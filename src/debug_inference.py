import torch
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

user_dir = 'DialogVED/'
model_dir = '/home/v-wchen2/data/dialogue/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt'

input_args = [
    '/home/v-wchen2/data/dialogue/pretrain/reddit/binary/finetune_sample',
    '--user-dir', user_dir,
    '--task', 'seq2seq_vae',
    '--batch-size', '16',
    '--beam', '1',
    '--num-workers', '4',
    '--gen-subset', 'valid',
    '--skip-invalid-size-inputs-valid-test',
    # '--path', '/home/v-wchen2/data/dialogue/checkpoints/_vae_bow_lm_standard_1800_16/checkpoint1.pt',
    # '--path', '/home/v-wchen2/data/dialogue/checkpoints/_seq2seq_bow_lm_1800_16/checkpoint1.pt'
    '--path', model_dir,
]

parser = options.get_generation_parser(args=input_args)
args = options.parse_args_and_arch(parser, input_args=input_args)
utils.import_user_module(args)

args.with_encoder_decoder_attn = True

if args.max_tokens is None and args.max_sentences is None:
    args.max_tokens = 12000
print(args)

use_cuda = torch.cuda.is_available() and not args.cpu

# Load dataset splits
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)

# set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary

# Load ensemble
print('| loading model(s) from {}'.format(args.path))
models, _model_args = checkpoint_utils.load_model_ensemble(
    args.path.split(':'),
    arg_overrides=eval(args.model_overrides),
    task=task,
)

from fairseq import tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
state = load_checkpoint_to_cpu(args.path)

args = state["args"]
if task is None:
    task = tasks.setup_task(args)

args.with_encoder_decoder_attn = True
args.with_cls_bow_logits = False
args.with_latent_bow_logits = True
args.with_mask_lm_logits = True
args.extend_latent_with_cls = True
ensemble = []
# build model for ensemble
model = task.build_model(args)
model.load_state_dict(state["model"], strict=True, args=args)
ensemble.append(model)


# optimize ensemble for generation
for model in models:
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()


align_dict = utils.load_align_dict(args.replace_unk)

# Load dataset (possibly sharded)
itr = task.get_batch_iterator(
    dataset=task.dataset(args.gen_subset),
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    ),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    num_shards=args.num_shards,
    shard_id=args.shard_id,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)

# Initialize generator
gen_timer = StopwatchMeter()
generator = task.build_generator(args)

sample = next(itr)

sample = utils.move_to_cuda(sample) if use_cuda else sample

prefix_tokens = None
if args.prefix_size > 0:
    prefix_tokens = sample['target'][:, :args.prefix_size]

# finally! success =.=
# fix all bugs on 2021.01.19 5:00
# congratulation to myself!


# hypos = task.inference_step(generator, models, sample, prefix_tokens)

# # into inference_step
# from fairseq.sequence_generator import EnsembleModel
# model = EnsembleModel(models)
#
# self = generator
# print(model.incremental_states)
# bos_token = None
#
# if not self.retain_dropout:
#     model.eval()
#
# # model.forward normally channels prev_output_tokens into the decoder
# # separately, but SequenceGenerator directly calls model.encoder
# encoder_input = {
#     k: v for k, v in sample['net_input'].items()
#     if k != 'prev_output_tokens'
# }
#
# src_tokens = encoder_input['src_tokens']
# src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
# input_size = src_tokens.size()
# # batch dimension goes first followed by source lengths
# bsz = input_size[0]
# src_len = input_size[1]
# beam_size = self.beam_size
#
# if self.match_source_len:
#     max_len = src_lengths.max().item()
# else:
#     max_len = min(
#         int(self.max_len_a * src_len + self.max_len_b),
#         # exclude the EOS marker
#         model.max_decoder_positions() - 1,
#     )
#
# # compute the encoder output for each beam
# encoder_outs = model.forward_encoder(encoder_input)
# new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
# new_order = new_order.to(src_tokens.device).long()
# encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)
#
# # initialize buffers
# scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
# scores_buf = scores.clone()
# tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
# tokens_buf = tokens.clone()
# tokens[:, 0] = self.eos if bos_token is None else bos_token
# attn, attn_buf = None, None
#
# # The blacklist indicates candidates that should be ignored.
# # For example, suppose we're sampling and have already finalized 2/5
# # samples. Then the blacklist would mark 2 positions as being ignored,
# # so that we only finalize the remaining 3 samples.
# blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask
#
# # list of completed sentences
# finalized = [[] for i in range(bsz)]
# finished = [False for i in range(bsz)]
# num_remaining_sent = bsz
#
# # number of candidate hypos per step
# cand_size = 2 * beam_size  # 2 x beam size in case half are EOS
#
# # offset arrays for converting between different indexing schemes
# bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
# cand_offsets = torch.arange(0, cand_size).type_as(tokens)
#
# # helper function for allocating buffers on the fly
# buffers = {}
#
# import math
#
# def buffer(name, type_of=tokens):  # noqa
#     if name not in buffers:
#         buffers[name] = type_of.new()
#     return buffers[name]
#
#
# def is_finished(sent, step, unfin_idx):
#     """
#     Check whether we've finished generation for a given sentence, by
#     comparing the worst score among finalized hypotheses to the best
#     possible score among unfinalized hypotheses.
#     """
#     assert len(finalized[sent]) <= beam_size
#     if len(finalized[sent]) == beam_size or step == max_len:
#         return True
#     return False
#
#
# def finalize_hypos(step, bbsz_idx, eos_scores):
#     """
#     Finalize the given hypotheses at this step, while keeping the total
#     number of finalized hypotheses per sentence <= beam_size.
#
#     Note: the input must be in the desired finalization order, so that
#     hypotheses that appear earlier in the input are preferred to those
#     that appear later.
#
#     Args:
#         step: current time step
#         bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
#             indicating which hypotheses to finalize
#         eos_scores: A vector of the same size as bbsz_idx containing
#             scores for each hypothesis
#     """
#     assert bbsz_idx.numel() == eos_scores.numel()
#
#     # clone relevant token and attention tensors
#     tokens_clone = tokens.index_select(0, bbsz_idx)
#     tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
#     assert not tokens_clone.eq(self.eos).any()
#     tokens_clone[:, step] = self.eos
#     attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None
#
#     # compute scores per token position
#     pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
#     pos_scores[:, step] = eos_scores
#     # convert from cumulative to per-position scores
#     pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
#
#     # normalize sentence-level scores
#     if self.normalize_scores:
#         eos_scores /= (step + 1) ** self.len_penalty
#
#     cum_unfin = []
#     prev = 0
#     for f in finished:
#         if f:
#             prev += 1
#         else:
#             cum_unfin.append(prev)
#
#     sents_seen = set()
#     for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
#         unfin_idx = idx // beam_size
#         sent = unfin_idx + cum_unfin[unfin_idx]
#
#         sents_seen.add((sent, unfin_idx))
#
#         if self.match_source_len and step > src_lengths[unfin_idx]:
#             score = -math.inf
#
#         def get_hypo():
#
#             if attn_clone is not None:
#                 # remove padding tokens from attn scores
#                 hypo_attn = attn_clone[i]
#             else:
#                 hypo_attn = None
#
#             return {
#                 'tokens': tokens_clone[i],
#                 'score': score,
#                 'attention': hypo_attn,  # src_len x tgt_len
#                 'alignment': None,
#                 'positional_scores': pos_scores[i],
#             }
#
#         if len(finalized[sent]) < beam_size:
#             finalized[sent].append(get_hypo())
#
#     newly_finished = []
#     for sent, unfin_idx in sents_seen:
#         # check termination conditions for this sentence
#         if not finished[sent] and is_finished(sent, step, unfin_idx):
#             finished[sent] = True
#             newly_finished.append(unfin_idx)
#     return newly_finished
#
#
# reorder_state = None
# batch_idxs = None
#
# step = 0
#
# if reorder_state is not None:
#     if batch_idxs is not None:
#         # update beam indices to take into account removed sentences
#         corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
#         reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
#     model.reorder_incremental_state(reorder_state)
#     encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
#
#
# # lprobs, avg_attn_scores = model.forward_decoder(
# #     tokens[:, :step + 1], encoder_outs, temperature=self.temperature,
# # )
#
# # into forward_decoder
# # temperature = self.temperature
# # tokens = tokens[:, :step + 1]
#
# print(model.models[0])
#
# import torch.nn.functional as F
#
# # finally debugging in decoder... this is really fucking...
# prev_output_tokens = tokens[:, :step + 1]
# incremental_state = model.incremental_states[model.models[0]]
# encoder_out = encoder_outs[0]
#
# self = model.models[0].decoder
# kwargs = {}
#
# if 'positions' in kwargs:
#     main_stream_pos_embed = self.embed_ap_token._forward(kwargs['positions'])
#     real_positions = kwargs['positions']
#     i_buckets_main_stream, i_bucket_relative_stream = \
#         self.cal_pretrain_relative_positions(real_positions)
# else:
#     # calculate main stream position embedding and relative positions
#     # from fairseq.modules import LearnedPositionalEmbedding
#     # the current LearnedPositionalEmbedding only return additional positions
#     main_stream_pos_embed, real_positions = self.embed_ap_token(
#         prev_output_tokens,
#         incremental_state=incremental_state,
#     ) if self.embed_ap_token is not None else None
#
#     if incremental_state is not None:
#         i_buckets_main_stream, i_bucket_relative_stream = None, None
#     else:
#         i_buckets_main_stream, i_bucket_relative_stream = \
#             self.cal_finetune_relative_positions(real_positions)
#
# # calculate predicting stream position embedding
# predicting_stream_pos_embed = self.embed_ap_token._forward(real_positions + 1)
#
# if incremental_state is not None:
#     # understand that, in the incremental decoding, only the last output is need for decoding
#     prev_output_tokens = prev_output_tokens[:, -1:]
#     if main_stream_pos_embed is not None:
#         main_stream_pos_embed = main_stream_pos_embed[:, -1:]
#
# x = self.embed_tokens(prev_output_tokens)
# if self.embed_scale is not None:
#     x *= self.embed_scale
#
# if main_stream_pos_embed is not None:
#     x += main_stream_pos_embed
#
# x = x.transpose(0, 1)
# attn = None
#
# inner_states = [x]
# if main_stream_pos_embed is None:
#     print('positions should be used to predict ngrams')
#     raise Exception()
#
# if self.embed_scale is not None:
#     ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
# else:
#     ngram_input_embed = self.ngram_input_embed.weight
# # ngram embedding
#
# if incremental_state is not None:
#     B = x.size(1)
#     ngram_masks = [
#         (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
#         for ngram in range(self.ngram)]
# else:
#     ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
#                    ngram in range(self.ngram)]
#
# self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
# ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None
#
# x = torch.cat([x] + ngram_masks, 0)
#
# if self.emb_layer_norm:
#     x = self.emb_layer_norm(x)
#
# x = F.dropout(x, p=self.dropout, training=self.training)
#
# # transform latent variable to latent key and value for multi-heads
# # skip_encoder_attn = True if self.training_mode == 'vae' else False
# # if self.training_mode == 'vae':
# # latent_context = encoder_out['latent_context']
# if self.with_encoder_decoder_attn:
#     _latent_context = None
# else:
#     _latent_context = self.transform_latent_context(encoder_out['z'])
#
# latent_context = _latent_context
# self = self.layers[0].ngram_self_attn
# query = x
# static_kv = False

# _model = model.models[0]

# decoder_out = list(_model.forward_decoder(
#     prev_output_tokens, encoder_out=encoder_out, incremental_state=model.incremental_states[_model],
# ))


# debugging ...
# hypos = task.inference_step(generator, models, sample, prefix_tokens)
# num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
