# debug fairseq-0.9.0 fairseq-train command
# 1) add import sys; print(sys.argv) in fairseq_cli/train.py
# 2) edit get_training_parser, get_parser function in fairseq/options.py
# then you can get complete args, and follow the code bellow to get complete debug process

from fairseq import options

# a complete fairseq debug code
import math
import collections

import torch
from fairseq import options, tasks, utils, checkpoint_utils, progress_bar
from fairseq.trainer import Trainer
from fairseq.data import iterators
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq_cli.train import get_training_stats, validate


user_dir = './DialogVED'
data_dir = '/home/v-wchen2/data/dialogue/pretrain/reddit/binary/finetune_sample'
# data_dir = '/home/v-wchen2/data/dialogue/finetune/dailydialog/binary'
# data_dir = '/home/v-bolunyao/wchen2/finetune/dailydialog/binary'
# model_dir = '/home/v-wchen2/data/dialogue/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt'
# model_dir = '/home/v-wchen2/data/dialogue/pretrain/reddit/checkpoints/_seq2seq_bow_lm_1800_16/checkpoint1.pt'
# model_dir = '/home/v-wchen2/data/dialogue/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt'
# model_dir = '/home/v-bolunyao/wchen2/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt'
model_dir = '/home/v-wchen2/data/dialogue/checkpoints/checkpoint2.pt'
tmp_dir = '/home/v-wchen2/data/dialogue/pretrain/reddit/tmp'

arch = 'ngram_transformer_prophet_seq2seq'
CRITERION = 'ved_loss'
task_name = 'ved_translate'

input_args = [
    data_dir,
    '--user-dir', user_dir,
    '--fp16',
    '--task', task_name,
    '--arch', arch,
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-6', '--clip-norm', '1.0',
    '--lr', '0.0003', '--lr-scheduler',
    'inverse_sqrt', '--min-lr', '1e-09', '--warmup-init-lr', '1e-07', '--warmup-updates', '5000',
    '--criterion', CRITERION,
    '--update-freq', '4', '--max-tokens', '2000', '--max-sentences', '4', '--num-workers', '1',
    # '--load-from-pretrained-model', model_dir,
    '--dropout', '0.1',
    '--label-smoothing', '0.1',
    '--attention-dropout', '0.1',
    '--activation-dropout', '0.0',
    '--weight-decay', '0.01',
    '--skip-invalid-size-inputs-valid-test',
    '--save-dir', tmp_dir,
    '--max-epoch', '10', '--keep-last-epochs', '10',
    '--tensorboard-logdir', tmp_dir,
    '--dataset-impl', 'mmap', '--empty-cache-freq', '64', '--seed', '1',
    '--skip-invalid-size-inputs-valid-test', '--distributed-no-spawn',
    '--log-interval', '200',
    '--seed', '2',
    # '--add-cls-to-source',
    '--mask-source',
    '--max-source-positions', '512',
    '--max-target-positions', '128',
    '--kl-loss-weight', '0.0',
    '--cls-bow-loss-weight', '0.0',
    '--latent-bow-loss-weight', '0.0',
    '--masked-lm-loss-weight', '0.0',
    # '--use-tfidf-weights',
    # '--tfidf-model-path', '/home/v-wchen2/data/dialogue/pretrain/reddit/tfidf_model',
    # '--tfidf-dictionary-path', '/home/v-wchen2/data/dialogue/pretrain/reddit/tfidf_dict',
]

parser = options.get_training_parser(args=input_args)
args = options.parse_args_and_arch(parser, input_args=input_args)
utils.import_user_module(args)
print(args)

torch.cuda.set_device('cuda:0')

task = tasks.setup_task(args)
task.load_dataset('valid', combine=False, epoch=0)

# single example
# print(task.datasets['valid'][0])

# batch examples
batch_size = 5
samples = [task.datasets['valid'][_] for _ in range(batch_size)]
# print(task.datasets['valid'].collater(samples))

print('[context] -> ' + task.source_dictionary.string(task.datasets['valid'][0]['source']))
print('[response] -> ' + task.source_dictionary.string(task.datasets['valid'][0]['target']))

print('[context] -> ' + task.source_dictionary.string(task.datasets['valid'][1]['source']))
print('[response] -> ' + task.source_dictionary.string(task.datasets['valid'][1]['target']))

# Build model and criterion
model = task.build_model(args)
criterion = task.build_criterion(args)
print(model)
print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
print('| num. model params: {} (num. trained: {})'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
))

args.load_from_pretrained_model = model_dir


# Build trainer
trainer = Trainer(args, task, model, criterion)
print('| training on {} GPUs'.format(args.distributed_world_size))
print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
    args.max_tokens,
    args.max_sentences,
))

# Load the latest checkpoint if one is available and restore the
# corresponding train iterator
extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

# Train until the learning rate gets too small
max_epoch = args.max_epoch or math.inf
max_update = args.max_update or math.inf
lr = trainer.get_lr()
train_meter = StopwatchMeter()
train_meter.start()
valid_subsets = args.valid_subset.split(',')

# into train function
update_freq = args.update_freq[epoch_itr.epoch - 1] \
    if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

# Initialize data iterator
itr = epoch_itr.next_epoch_itr(
    fix_batches_to_gpus=args.fix_batches_to_gpus,
    shuffle=(epoch_itr.epoch >= args.curriculum),
)
itr = iterators.GroupedIterator(itr, update_freq)
progress = progress_bar.build_progress_bar(
    args, itr, epoch_itr.epoch, no_progress_bar='simple',
)

extra_meters = collections.defaultdict(lambda: AverageMeter())
valid_subsets = args.valid_subset.split(',')
max_update = args.max_update or math.inf

model.train()
criterion.train()

from tqdm import tqdm

end = False

for i, samples in tqdm(enumerate(progress, start=epoch_itr.iterations_in_epoch)):
    for j in range(len(samples)):
        if i == 401 and j == 3:
            print('fucking.......')
            # end = True
            # break
        # if end:
        #     break
        sample = utils.move_to_cuda(samples[j])
        decoder_out, encoder_out = model(**sample['net_input'], return_all_hiddens=False)

for samples in progress:
    break

sample = samples[0]
sample = utils.move_to_cuda(sample)
# nsentences = sample['nsentences']
ntokens = sample['ntokens']
src_tokens = sample['net_input']['src_tokens']
src_lengths = sample['net_input']['src_lengths']
# _sentence_positions = sample['net_input']['_sentence_positions']
# _role_positions = sample['net_input']['_role_positions']
# _relative_position_token = sample['net_input']['_relative_position_token']
# _relative_position_sentence = sample['net_input']['_relative_position_sentence']
# _src_masked_indices = sample['net_input']['_src_masked_indices']
prev_output_tokens = sample['net_input']['prev_output_tokens']

target = sample['target']
encoder_out = model.encoder(**sample['net_input'])


# for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
#     log_output = trainer.train_step(samples)
#     if log_output is None:
#         continue
#
#     # log mid-epoch stats
#     stats = get_training_stats(trainer)
#     for k, v in log_output.items():
#         if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
#             continue  # these are already logged above
#         if 'loss' in k or k == 'accuracy':
#             extra_meters[k].update(v, log_output['sample_size'])
#         else:
#             extra_meters[k].update(v)
#         stats[k] = extra_meters[k].avg
#     progress.log(stats, tag='train', step=stats['num_updates'])
#
#     # ignore the first mini-batch in words-per-second and updates-per-second calculation
#     if i == 0:
#         trainer.get_meter('wps').reset()
#         trainer.get_meter('ups').reset()
#
#     num_updates = trainer.get_num_updates()
#     if (
#             not args.disable_validation
#             and args.save_interval_updates > 0
#             and num_updates % args.save_interval_updates == 0
#             and num_updates > 0
#     ):
#         valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
#         checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
#
#     if num_updates >= max_update:
#         break


# debugging decoder...
sample = samples[-1]
model.train()
criterion.train()
sample = utils.move_to_cuda(sample)

# encoder
# execute forward method defined in model
decoder_out, encoder_out = model(**sample['net_input'], return_all_hiddens=False)

# n-gram predicting stream
logits_list = decoder_out[0]

self = criterion

# decoder
import torch.nn.functional as F

nsentences = sample['nsentences']
ntokens = sample['ntokens']
src_tokens = sample['net_input']['src_tokens']
src_lengths = sample['net_input']['src_lengths']
prev_output_tokens = sample['net_input']['prev_output_tokens']
src_masked_indices = sample['net_input']['src_masked_indices']
target = sample['target']

incremental_state = None
kwargs = sample


# reddit
# model = model


