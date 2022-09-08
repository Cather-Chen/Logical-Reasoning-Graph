from feature_adj_utils import load_and_cache_examples,convert_examples_to_features, processors
from tokenization_dagn import arg_tokenizer
from my_network_sw import MyHGAT
import argparse
import glob
import logging
import os
import warnings
warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaForMultipleChoice,
    get_linear_schedule_with_warmup,
)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from graph_building_blocks.argument_set_punctuation_v4 import punctuations
with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
    relations = json.load(f)  # key: relations, value: ignore

logger = logging.getLogger(__name__)
config_class = RobertaConfig
tokenizer_class = RobertaTokenizer
model_class = MyHGAT

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    #在tensorboardX上记录
    if args.local_rank in [-1, 0]:
        str_list = str(args.output_dir).split('/')
        tb_log_dir = os.path.join('summaries', str_list[-1])
        tb_writer = SummaryWriter(tb_log_dir)

    #train_batch_size = 每个gpu上的batchsize * gpu个数
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=args.number_workers, pin_memory=torch.cuda.is_available())

    #max_steps 代表迭代次数上限，<0代表无上限
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # args.num_train_epochs = 3（default）

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]  #bias和norm层的weight不优化
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    exec('args.adam_betas = ' + args.adam_betas)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)
    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    ) # 优化一次需要的样本batch_size
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    def evaluate_model(train_preds, train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc,test=False, save_result=False):
        """ 当前模型在dev上测试，若acc提高了，则更新模型参数 """
        train_preds = np.argmax(train_preds, axis=1)   #softmax输出最可能的选项
        train_acc = simple_accuracy(train_preds, train_label_ids)
        train_preds = None
        train_label_ids = None
        results = evaluate(args, model, tokenizer, test, save_result)
        logger.info(
            "train acc: %s, dev acc: %s, loss: %s, global steps: %s",
            str(train_acc),
            str(results["eval_acc"]),
            str(results["eval_loss"]),
            str(global_step),
        )
        tb_writer.add_scalar("training/acc", train_acc, global_step)
        for key, value in results.items():
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        if results["eval_acc"] > best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("achieve BEST dev acc: %s at global step: %s",
                        str(best_dev_acc),
                        str(best_steps)
            )

            # if args.do_test:
            #     results_test = evaluate(args, model, tokenizer, test=True)
            #     for key, value in results_test.items():
            #         tb_writer.add_scalar("test_{}".format(key), value, global_step)
            #     logger.info(
            #         "test acc: %s, loss: %s, global steps: %s",
            #         str(results_test["eval_acc"]),
            #         str(results_test["eval_loss"]),
            #         str(global_step),
            #     )
            # save best dev acc model
            # output_dir = os.path.join(args.output_dir, "checkpoint-best")
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_vocabulary(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
            with open(txt_dir, 'w') as f:
                rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
                f.write(rs)
                tb_writer.add_text('best_results', rs, global_step)
            # 每提高一次验证集表现则储存一次当前step作为best_step,更新best_dev_acc，并保存到主文件夹

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    def save_model(args, model, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))   #在第几步时储存的checkpoints
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_vocabulary(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)


    # 初始化
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:   # 生成10个epoch迭代器
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)   # 数据分布到device上
            input_ids =batch[0]
            attention_mask =batch[1]
            argument_bpe_ids =batch[4]
            punct_bpe_ids =batch[6]
            keytokensids =batch[7]
            keymask =batch[8]
            key_segid =batch[11]
            SVO_ids =batch[9]
            SVO_mask =batch[10]
            adj_SVO =batch[13]
            labels = batch[12]
            passage_mask=batch[2]
            question_mask=batch[3]
            # token_type = batch[-1]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, argument_bpe_ids=argument_bpe_ids,
                            punct_bpe_ids=punct_bpe_ids, keytokensids=keytokensids, keymask=keymask,
                            key_segid=key_segid, SVO_ids=SVO_ids, SVO_mask=SVO_mask, adj_SVO=adj_SVO,
                            labels=labels, passage_mask=passage_mask, question_mask=question_mask)

            # outputs = model(input_ids,attention_mask = attention_mask,labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            ################# work only gpu = 1 ######################
            if train_preds is None:
                train_preds = logits.detach().cpu().numpy()
                train_label_ids = labels.detach().cpu().numpy()
            else:
                train_preds = np.append(train_preds, logits.detach().cpu().numpy(), axis=0)
                train_label_ids = np.append(train_label_ids, labels.detach().cpu().numpy(), axis=0)
            ###########################################################


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training!!!!
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    # 梯度剪切，规定了最大不能超过的max_norm
            else:
                loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # for name, parms in model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度积累三次后优化下降
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1   # 全局steps
                # 每args.logging_steps后输出一次evaluate结果
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids,
                                                                                                           tb_writer, args, model, tokenizer, best_steps, best_dev_acc)
                    tb_writer.add_scalar("training/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("training/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s, average acc: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),  # logging_steps次的平均loss
                        str(train_acc),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds,
        train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc, test=False, save_result=True)
        save_model(args, model, tokenizer)
        tb_writer.close()

    # 每logging_steps(200)次测试一次模型，不足200次的最终训练结果也保存一次
    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, test=False, save_result=False, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args,tokenizer,arg_tokenizer, evaluate=not test, test=test)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=args.number_workers, pin_memory=torch.cuda.is_available())

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        example_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                input_ids = batch[0]
                attention_mask = batch[1]
                argument_bpe_ids = batch[4]
                punct_bpe_ids = batch[6]
                keytokensids = batch[7]
                keymask = batch[8]
                key_segid = batch[11]
                SVO_ids = batch[9]
                SVO_mask = batch[10]
                adj_SVO = batch[13]
                labels = batch[12]
                passage_mask = batch[2]
                question_mask = batch[3]
                # token_type=batch[-1]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,argument_bpe_ids= argument_bpe_ids,
                                punct_bpe_ids=punct_bpe_ids, keytokensids=keytokensids, keymask=keymask,
                                 key_segid=key_segid,SVO_ids=SVO_ids, SVO_mask=SVO_mask,adj_SVO= adj_SVO,
                                labels=labels,passage_mask=passage_mask,question_mask=question_mask)
                # outputs = model(input_ids,attention_mask = attention_mask,labels=labels)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            # writer.write("model           =%s\n" % str(args.model_name_or_path))
            # writer.write(
            #     "total batch size=%d\n"
            #     % (
            #         args.per_gpu_train_batch_size
            #         * args.gradient_accumulation_steps
            #         * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
            #     )
            # )
            # writer.write("train num epochs=%d\n" % args.num_train_epochs)
            # writer.write("fp16            =%s\n" % args.fp16)
            # writer.write("max seq length  =%d\n" % args.max_seq_length)
            if not test:
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
    if test:
        return results, preds
    else:
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--task_name",
        default="reclor",
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", default=True,help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", default=True, help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", default=True, help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true",default=True, help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=3,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--adam_betas', default='(0.9, 0.98)', type=str, help='betas for Adam optimizer')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_clip_grad_norm", action="store_true", default=True, help="whether not to clip grad norm")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=15, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup over warmup ratios.")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=800, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true",default='Checkpoints/reclor/roberta-large', help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--number_workers",type=int,default=4,help='number of workers')

    args = parser.parse_args()
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        # os.listdir(xx): 返回指定路径下的文件和文件夹列表。
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),  # args.local_rank = -1 代表无分布式训练
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    if args.local_rank not in [-1, 0]:  # -1代表无分布式，0 代表非主进程，先暂停!
        torch.distributed.barrier()

    config = config_class.from_pretrained(args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    max_rel_id = int(max(relations.values()))
    feature_dim_list = [config.hidden_size]*2

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        max_rel_id=max_rel_id,
        feature_dim_list=feature_dim_list,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    if args.local_rank == 0:   #主进程至此完成缓存，设置barrier，同时释放所有进程
        torch.distributed.barrier()

    # model = MyHGAT(config, max_rel_id, feature_dim_list, device= args.device)
    if args.local_rank == 0:   #主进程至此完成缓存，设置barrier，同时释放所有进程
        torch.distributed.barrier()
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer,arg_tokenizer, evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint,
                                                max_rel_id=max_rel_id,
                                                feature_dim_list=feature_dim_list,
                                                device=args.device,)
            # model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

if __name__ == "__main__":
    main()