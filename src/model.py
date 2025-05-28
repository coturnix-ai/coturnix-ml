import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
return results
import numpy as np
import tensorflow.compat.v1 as tf
from functools import partial
from data.encoders import encode
import random
import re
import logging
from itertools import cycle
from utils import natural_sort
def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        
        
        ### IN USE ###
        
        def _get_number_of_documents(filename):
            # extracts number of files from a filename formatted "<name>_<num_documents>.tfrecords."
            # if no pattern is matched, returns None
            match = re.search("_(\d{1,}).tfrecords$", filename)
            return int(match.group(1)) if match is not None else match
        
        
        def _get_number_of_documents_by_iteration(filename):
            # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
            # this could be very slow.
            logging.warning(
                "inputs/sequential_input() found no metadata found in filename - iterating through first tfrecord to find global length")
            count = 0
            for item in tf.io.tf_record_iterator(filename):
                count += 1
            return count
        
        
        def _get_skip_index(all_files, n_batches):
            prev_cumsum = 0
            cumsum = 0
            global_n_documents = None
            for count, f in cycle(enumerate(all_files)):
                prev_cumsum = cumsum
                if _get_number_of_documents(f) is not None:
                    cumsum += _get_number_of_documents(f)
                elif global_n_documents is None:
                    global_n_documents = _get_number_of_documents_by_iteration(f)
                    cumsum += global_n_documents
                else:
                    cumsum += global_n_documents
                if cumsum == n_batches:
                    remainder = 0
                    skip_idx = count + 1
                elif cumsum > n_batches:
                    remainder = n_batches - prev_cumsum
                    skip_idx = count
                    break
            return skip_idx, remainder
        
        
        def _parse_function(example_proto):
            features = {
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return tf.sparse.to_dense(parsed_features["text"], parsed_features["text"].dense_shape[0])
        
        
        def autoregressive_sample_text(params, x):
            vals1 = x[:params["n_ctx"]]
            vals2 = x[1:params["n_ctx"] + 1]
        
            vals1 = tf.reshape(vals1, [params["n_ctx"]])
            vals2 = tf.reshape(vals2, [params["n_ctx"]])
            vals1 = tf.cast(vals1, dtype=tf.int32)
            vals2 = tf.cast(vals2, dtype=tf.int32)
            return vals1, vals2
        
        
        def sequential_input(params, global_step=None, eval=False):
            """
            Input fn that reads tfrecords encoded with a fixed chunk size (== n_ctx + 1), and that either:
        
                - has the number of documents for each tfrecord file encoded in the title in the format
                  <name>_<n_documents>.tfrecords.
        
                  OR
        
                - has a fixed number of documents per tfrecord file.
        
            If the glob pattern above isn't matched, we assume that each document has the same number of samples as the first tfrecord read.
            If this isn't the case, it may result in errors, or some samples being missed.
        
            This means we can calculate the number of samples we've seen so far using the global step,
            and can use dataset.skip() to iterate through the list of filenames, as opposed to the whole dataset, which is incredibly inefficient.
        
            If training is starting and stopping often, as with TPU pre-emption, reading the whole dataset sequentially appears to improve model
            performance, as it results in less repeated data.
            """
            if not eval:
                assert global_step is not None
            logging.warning(
                "Changing batch size with sequential_input() will result in some data being skipped or repeated. Please ensure your batch size stays constant throughout training.")
            batch_size = params['eval_batch_size' if eval else 'train_batch_size']
        
            filenames = []
            for dataset_config in params['dataset_configs'].values():  # iterate through each dataset and read params
                path_key = 'path' if not eval else 'eval_path'
                path = dataset_config[path_key]
                filenames.extend(
                    tf.io.gfile.glob(path))  # then glob all files that fit the pattern specified in dataset_configs
        
            filenames = natural_sort(filenames)
            shuffle_filenames = params.get("shuffle_input_filenames", True)
            if shuffle_filenames:
                seed = params.get('seed', 1)  # shuffle deterministically
                random.seed(seed)
                random.shuffle(filenames)
        
            dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat()  # repeat filenames to infinity
        
            if not eval:
                # skip forward first in the filenames list, then skip the remaining amount in the parsed tfrecords files
                skip_idx, remainder = _get_skip_index(filenames, n_batches=global_step * params[
                    "train_batch_size"])  # TODO: fix for > 1 epoch
                dataset = dataset.skip(skip_idx)  # skip to skip idx
        
                # read tfrecord examples and skip remainder
                dataset = dataset.apply(tf.data.TFRecordDataset)
                dataset = dataset.skip(remainder)
            else:
                # shuffle filenames if in eval mode
                dataset = dataset.shuffle(len(filenames))
                dataset = dataset.apply(tf.data.TFRecordDataset)
        
            # parse the tokenized data from the tfrecord files and shuffle
            dataset = dataset.map(_parse_function, num_parallel_calls=1)
            dataset = dataset.map(partial(autoregressive_sample_text, params), num_parallel_calls=1)
        
            # batch data and repeat to infinity
            dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(params["iterations"] * 2)
            return dataset.repeat()
        
        
        def pred_input(params, logger, enc=None,
                       path_to_prompt=""):
            unicorns = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
                       "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
                       "researchers was the fact that the unicorns spoke perfect English."
        
            text = unicorns if path_to_prompt == "" else open(path_to_prompt, "r").read()
            tokens = encode(enc, text)
        
            if len(tokens) > params["n_ctx"]:
                logger.info("The length of your input prompt is longer than the model's context length - truncating input.")
                tokens = tokens[len(tokens) - params["n_ctx"]:]
            if len(tokens) < params["n_ctx"]:
                tokens = tf.pad(tokens, [[0, params["n_ctx"] - len(tokens)]], constant_values=params["padding_id"])
            t = tf.broadcast_to(tokens, [params["batch_size"], params["n_ctx"]])
            dataset = tf.data.Dataset.from_tensors(t)
        
            def _dummy_labels(x):
                return x, x
        
            dataset = dataset.map(_dummy_labels)
            return dataset
        
        
        def handle_pred_output(predictions, logger, enc, params, out_name="test"):
            with tf.gfile.Open(f"{out_name}.txt", "w") as f:
                for i, p in enumerate(predictions):
                    p = p["outputs"]
        
                    # remove eos + padding ids from output
                    idx = np.argmax(p == params['eos_id'])
                    if idx > 0:
                        p = p[:idx]
                    idx = np.argmax(p == params['padding_id'])
                    if idx > 0:
                        p = p[:idx]
        
                    text = enc.decode(p)
                    f.write("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
                    f.write(text)
                    f.write("\n" + "=" * 80 + "\n")
        
                    logger.info("=" * 40 + " SAMPLE " + str(i) + " " + "=" * 40 + "\n")
                    logger.info(text)
                    logger.info("\n" + "=" * 80 + "\n")
        
        
        ### DEPRECATED ###
        
        def generic_text(params, eval=False, sample_text_fn=None, **kwargs):
            logging.warning("DEPRECATION WARNING: generic_text will be phased out in future versions.")
            i = 0 if not eval else 1
        
            weights = []
            datasets = []
        
            for dataset in params["datasets"]:
                dataset_id, stitch, datatype, weight = dataset
        
                assert dataset_id in params[
                    'dataset_configs'], f'Unknown dataset id {dataset_id} given. Please make sure your dataset ids contain that configuration'
                dataset_config = params['dataset_configs'][dataset_id]
        
                path_key = 'path' if not eval else 'eval_path'
                path = dataset_config[path_key]
        
                datasets.append(text_dataset(
                    tf.io.gfile.glob(path),
                    params,
                    stitch=stitch,
                    datatype=datatype,
                    batch=False,
                    sample_text_fn=sample_text_fn
                ))
        
                weights.append(weight)
        
            batch_size = params['eval_batch_size' if eval else 'train_batch_size']
        
            seed = params.get('seed', None)
            dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights, seed=seed)
            dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(params["iterations"] * 2)
            return dataset
        
        
        def text_dataset(files, params, stitch, datatype, batch=True, sample_text_fn=None):
            seed = params.get('seed', None)
            deterministic = seed is not None
            num_parallel_calls = 1 if deterministic else tf.data.experimental.AUTOTUNE
        
            dataset = tf.data.Dataset.from_tensor_slices(files)
        
            if deterministic:
                dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4)
            else:
                dataset = dataset.apply(
                    tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))
        
            if "documents" in datatype:
                def _parse_function(example_proto):
                    features = {
                        # "hash": tf.VarLenFeature(tf.string),
                        "text": tf.VarLenFeature(tf.int64)
                    }
                    parsed_features = tf.parse_single_example(example_proto, features)
                    return parsed_features["text"], parsed_features["text"].dense_shape[0]
            else:
                def _parse_function(example_proto):
                    features = {
                        "text": tf.VarLenFeature(tf.int64)
                    }
                    parsed_features = tf.parse_single_example(example_proto, features)
                    return parsed_features["text"]  # Assuming the text is not sparse
        
            dataset = dataset.map(_parse_function, num_parallel_calls=1)
        
            # Subsample method
            if "documents" in datatype:
                # Since samples can be less than the correct length, and TPUs don't like variable lengths, this function stitches together enough samples
                # to have a text at least 1024 tokens long. For this to work the stitch parameter must be correctly tuned so that
                # stitch * min(characters_in_text) >= amount
                def _stitch_text(x, y):
                    x = tf.sparse.to_dense(x)
        
                    def _get_x(i):
                        return tf.gather(x[i], tf.range(y[i]))
        
                    out = _get_x(0)
                    eos_id = params['eos_id']
        
                    for i in range(1, stitch):
                        out = tf.concat([out, [eos_id], _get_x(i)], axis=0)  # text1<|endoftext|>text2
        
                    return out
        
                # Hack-y way to stitch together multiple texts
        
                dataset = dataset.shuffle(1000 * stitch, seed=seed).batch(stitch, drop_remainder=True).map(_stitch_text,
                                                                                                           num_parallel_calls=num_parallel_calls)
        
                # Sample 1024(+1) tokens from the stitched together text
                is_random_documents = datatype == "documents_random"
                if sample_text_fn is not None:
                    _sample_text = partial(sample_text_fn, random_documents=is_random_documents)
                else:
                    _sample_text = autoregressive_sample_text_random_documents if is_random_documents else autoregressive_sample_text
                    _sample_text = partial(_sample_text, params)
        
                dataset = dataset.map(_sample_text, num_parallel_calls=num_parallel_calls)
        
            if batch:
                dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)
        
            dataset = dataset.repeat()
        
            return dataset
        
        
        def autoregressive_sample_text_random_documents(params, x):
            seed = params.get('seed', None)
            s = tf.size(x)
            r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32, seed=seed)
            r1 = tf.range(r, r + params["n_ctx"])
            r2 = tf.range(r + 1, (r + 1) + params["n_ctx"])
            r1 = tf.reshape(r1, [params["n_ctx"]])  # Somehow, this makes the compiler happy
            r2 = tf.reshape(r2, [params[
                                     "n_ctx"]])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
            vals1 = tf.gather(x, r1)
            vals2 = tf.gather(x, r2)
        
            vals1 = tf.reshape(vals1, [params["n_ctx"]])
            vals2 = tf.reshape(vals2, [params["n_ctx"]])
            vals1 = tf.cast(vals1, dtype=tf.int32)
            vals2 = tf.cast(vals2, dtype=tf.int32)
            return vals1, vals2
        
        
        def mlm_sample_text(params, x, random_documents=False):
            seed = params.get('seed', None)
            ctx_len = params["n_ctx"]
            assert 'mlm_mask_id' in params, 'the key `mlm_mask_id` must be set on your config to do masked language model training, specifying the id of the reserved mask token'
        
            mask_id = params['mlm_mask_id']
            cls_token_id = params.get('mlm_cls_token_id', None)
            num_tokens = params.get('n_vocab', None)
        
            mask_ignore_ids = set(params.get('mlm_mask_ignore_ids', []))
            mask_ignore_ids.add(cls_token_id)
        
            mask_prob = params.get('mlm_mask_prob', 0.15)
            same_token_prob = params.get('mlm_same_token_prob', 0.10)
            random_token_prob = params.get('mlm_random_token_prob', 0.)
        
            seq_len = ctx_len if cls_token_id is None else (ctx_len - 1)
        
            if random_documents:
                s = tf.size(x)
                r = tf.random.uniform([], maxval=(s - seq_len), dtype=tf.dtypes.int32, seed=seed)
                r1 = tf.range(r, r + seq_len)
                r1 = tf.reshape(r1, [seq_len])
                features = tf.gather(x, r1)
            else:
                features = x[:seq_len]
        
            # add cls token id if specified by `mlm_cls_token_id`
            if cls_token_id is not None:
                features = tf.pad(features, [[1, 0]], constant_values=cls_token_id)
        
            features = tf.cast(features, dtype=tf.int32)
            shape = features.shape
        
            # determine which tokens are mask-able
            can_mask = tf.not_equal(features, 0)
            for ignore_id in mask_ignore_ids:
                can_mask &= tf.not_equal(features, ignore_id)
        
            # generate boolean mask for masking ids
            mask_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed), mask_prob)
            mask_mask &= can_mask
        
            # generate mask for actually replacing the tokens, for allowing a small number of tokens to stay the same
            replace_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed),
                                   1 - same_token_prob)
        
            # randomly replace some tokens with random tokens before masking
            if random_token_prob > 0:
                random_token_mask = tf.less(tf.random.uniform(shape, minval=0., maxval=1., dtype=tf.float32, seed=seed),
                                            random_token_prob)
                random_tokens = tf.random.uniform(shape, minval=1, maxval=num_tokens, dtype=tf.dtypes.int32, seed=seed)
        
                # make sure random tokens do not include illegal token ids specified by `mlm_mask_ignore_ids`
                random_can_mask = tf.not_equal(random_tokens, 0)
                for ignore_id in mask_ignore_ids:
                    random_can_mask &= tf.not_equal(random_tokens, ignore_id)
        
                features = tf.where(random_token_mask & random_can_mask, random_tokens, features)
        
            # mask the tokens
            mask_tokens = tf.ones(shape, dtype=tf.int32) * mask_id
            masked_features = tf.where(mask_mask & replace_mask, mask_tokens, features)
        
            # labels will be set to 0 for all non-masked tokens
            labels = tf.where(mask_mask, tf.zeros(shape, dtype=tf.int32), features)
        
            masked_features, labels = map(lambda t: tf.reshape(t, [ctx_len]), (masked_features, labels))
            return masked_features, labels
