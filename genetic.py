import random
import tqdm
import shutil
import numpy as np
import copy
from transformers.models.bert.modeling_bert import BertSelfAttention
import multiprocessing as mp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForSequenceClassification,
)
from torch.optim import AdamW

from transformers import BitsAndBytesConfig
import gc
import time
import pickle as pkl
from evaluate import load


def calc_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def calc_model_score_and_time(model, valid_dataloader, metric):
    metric = load("glue", "cola")

    sigmoid = nn.Sigmoid()
    device = str(model.device)
    model.eval()
    predictions = []
    actual_labels = []
    beg_time = time.time()
    with torch.no_grad():
        for batch in tqdm.tqdm(valid_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = (
                sigmoid(
                    model(input_ids=input_ids, attention_mask=attention_mask).logits
                ).squeeze()
                > 0.5
            )
            preds = outputs.cpu().numpy()
            predictions.extend(preds.tolist())
            actual_labels.extend(labels.tolist())

    res_time = time.time() - beg_time

    res_time = 1

    metric_res = metric.compute(predictions=predictions, references=actual_labels)
    metric_res = metric_res[sorted(metric_res.keys())[0]]

    return metric_res, res_time


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    """
    model = model.to('cpu')

    del model
    gc.collect()
    torch.cuda.empty_cache() 
    """
    copyOfModel.bert.encoder.layer = newModuleList

    return copyOfModel


def pruning(
    model_prune, max_prune_frac, max_attention_layers_prune_num, train_dataloader
):
    train_dataset = train_dataloader.dataset

    device = str(model_prune.device)

    prune_frac = 2**max_prune_frac / 100

    bert_prune = model_prune.bert

    example_inputs = [train_dataset[i] for i in range(100)]

    example_inputs = list(map(lambda x: {'input_ids': x['input_ids'].unsqueeze(0).to(device), 'attention_mask': x['attention_mask'].unsqueeze(0).to(device)}, example_inputs))[0]

    #outputs = model(**example_inputs)
    #last_hidden_states = outputs.last_hidden_state

    imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
    base_macs, base_params = tp.utils.count_ops_and_params(bert_prune, example_inputs)
    num_heads = {}

    # All heads should be pruned simultaneously, so we group channels by head.
    for m in bert_prune.modules():
        if isinstance(m, BertSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    pruner = tp.pruner.MetaPruner(
        bert_prune, 
        example_inputs, 
        global_pruning=False,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=prune_frac,
        num_heads=num_heads,
        prune_head_dims=False,
        prune_num_heads=False,
        head_pruning_ratio=0.3,
        output_transform=lambda out: out.pooler_output.sum(),
        ignored_layers=[bert_prune.pooler, bert_prune.embeddings],
    )

    for g in pruner.step(interactive=True):
        g.prune()

    # Modify the attention head size and all head size after pruning
    for m in bert_prune.modules():
        if isinstance(m, BertSelfAttention):
            m.num_attention_heads = pruner.num_heads[m.query]
            m.attention_head_size = m.query.out_features // m.num_attention_heads
            m.all_head_size = m.query.out_features

    attention_layers_prune_num = random.randint(0, max_attention_layers_prune_num)

    num_layers_to_keep = max(
        len(model_prune.bert.encoder.layer) - attention_layers_prune_num, 2
    )

    model_prune = deleteEncodingLayers(model_prune, num_layers_to_keep)

    return model_prune


def distillation(model_dist, model_teacher, train_epoch_per_iter, train_dataloader, lr):

    teacher_device = str(model_teacher.device)
    device = str(model_dist.device)

    optimizer = AdamW(model_dist.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader.dataset) * train_epoch_per_iter
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    sigmoid = nn.Sigmoid()

    for _ in tqdm.tqdm(range(train_epoch_per_iter)):

        model_dist.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model_dist.forward(
                input_ids=input_ids, attention_mask=attention_mask
            )
            encoder_outputs = outputs.hidden_states[-1]
            model_outputs = outputs.logits.squeeze()

            with torch.no_grad():

                outputs_big = model_teacher.forward(
                    input_ids=input_ids.to(teacher_device),
                    attention_mask=attention_mask.to(teacher_device),
                )
                encoder_outputs_big = outputs_big.hidden_states[-1].to(device)
                model_outputs_big = outputs_big.logits.squeeze().to(device)

            loss = (
                nn.BCEWithLogitsLoss(reduction="mean")(model_outputs, labels)
                + (
                    1
                    - nn.CosineSimilarity(dim=-1)(
                        sigmoid(model_outputs), sigmoid(model_outputs_big)
                    ).mean()
                )
                / 2
                + nn.MSELoss()(encoder_outputs, encoder_outputs_big)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

    model_teacher = model_teacher.to("cpu")
    del model_teacher
    gc.collect()
    torch.cuda.empty_cache()

    return model_dist


def simple_train(model, train_epoch_per_iter, train_dataloader, lr):

    device = str(model.device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader.dataset) * train_epoch_per_iter
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    sigmoid = nn.Sigmoid()

    for _ in tqdm.tqdm(range(train_epoch_per_iter)):

        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
            model_outputs = outputs.logits.squeeze()

            loss = nn.BCEWithLogitsLoss(reduction="mean")(model_outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    return model


def quantization(model_path, quant_method, device):

    if quant_method == "4":

        bnb_config = BitsAndBytesConfig(  # 4
            load_in_4bit=True,  # Гибридный режим
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant_method == "8":

        bnb_config = BitsAndBytesConfig(  # 8
            load_in_8bit=True,  # Гибридный режим
            llm_int8_skip_modules=["classifier", "pooler"],
        )
    else:
        bnb_config = None

    # bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 16
        quantization_config=bnb_config,
        device_map=device,
    )

    return model


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


def process_model(
    i,
    gpu_id,
    pop_size,
    a,
    b,
    c,
    max_prune_frac,
    max_attention_layers_prune_num,
    pruning_prob,
    quantization_prob,
    distillation_prob,
    train_epoch_per_iter,
    possible_quantizations,
    lr,
    train_dataloader,
    valid_dataloader,
    metric,
    pick_pairing_probs,
):
    model_path = f"./test_model_{i}"
    device = f"cuda:{gpu_id}"
    cur_model = BertForSequenceClassification.from_pretrained(
        model_path, device_map=device
    )

    is_pruning = random.uniform(0, 1) < pruning_prob

    if is_pruning:
        cur_model = pruning(
            cur_model, max_prune_frac, max_attention_layers_prune_num, train_dataloader
        )

    is_distillation = random.uniform(0, 1) < distillation_prob

    if is_distillation:
        pair_ind = np.random.choice(
            list(range(pop_size)), size=1, replace=False, p=pick_pairing_probs
        )[0]
        teacher_path = "./best_score_model"

        pair = BertForSequenceClassification.from_pretrained(
            teacher_path, device_map=device
        )
        cur_model = distillation(
            cur_model, pair, train_epoch_per_iter, train_dataloader, lr
        )
    else:
        cur_model = simple_train(cur_model, train_epoch_per_iter, train_dataloader, lr)

    dic = cur_model.config.to_dict()
    dic["num_hidden_layers"] = len(cur_model.bert.encoder.layer)
    dic["intermediate_size"] = cur_model.bert.encoder.layer[
        0
    ].intermediate.dense.out_features
    conf = BertConfig.from_dict(dic)

    is_quantization = random.uniform(0, 1) < quantization_prob

    if is_quantization:

        cur_model.save_pretrained(f"./test_model_{pop_size + i}")
        conf.save_pretrained(f"./test_model_{pop_size + i}")

        cur_model.save_pretrained(f"./test_model_{i}_quant")
        conf.save_pretrained(f"./test_model_{i}_quant")
        cur_model = cur_model.to("cpu")
        del cur_model

        quant_method = random.choice(possible_quantizations)
        cur_model_quant = quantization(f"./test_model_{i}_quant", quant_method, device)
        shutil.rmtree(f"./test_model_{i}_quant")

        cur_model_size = calc_model_size(cur_model_quant)
        cur_model_score, cur_model_inference_time = calc_model_score_and_time(
            cur_model_quant, valid_dataloader, metric
        )
        if quant_method != "8":
            cur_model_quant = cur_model_quant.to("cpu")
        del cur_model_quant
    else:
        quant_method = None
        cur_model_size = calc_model_size(cur_model)
        cur_model_score, cur_model_inference_time = calc_model_score_and_time(
            cur_model, valid_dataloader, metric
        )
        cur_model.save_pretrained(f"./test_model_{pop_size + i}")
        conf.save_pretrained(f"./test_model_{pop_size + i}")
        cur_model = cur_model.to("cpu")

        del cur_model

    cur_model_fitness = cur_model_score**a / (
        cur_model_size**b * cur_model_inference_time**c
    )
    gc.collect()
    torch.cuda.empty_cache()

    return (
        cur_model_fitness,
        cur_model_score,
        quant_method,
        cur_model_size,
        cur_model_inference_time,
    )


if __name__ == "__main__":

    with open("data.pkl", "rb") as f:
        dataset = pkl.load(f)

    # Разделение данных
    train_data = dataset["train"][:]
    valid_data = dataset["validation"][:]

    train_texts = train_data["sentence"].tolist()
    train_labels = train_data["label"].tolist()

    valid_texts = valid_data["sentence"].tolist()
    valid_labels = valid_data["label"].tolist()

    metric = load("glue", "cola")

    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, num_labels=1
    )
    model = BertForSequenceClassification.from_pretrained(
        "./test_model_bert",
        # problem_type="multi_label_classification",
        config=config,
        device_map="cuda:0",
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, 500)
    valid_dataset = TextClassificationDataset(valid_texts, valid_labels, tokenizer, 500)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32)

    max_epoch = 5
    early_stop_epoch = 2
    pop_size = 10
    max_proc = 10
    a = 1
    b = 15
    c = 1
    max_prune_frac = 0.5
    max_attention_layers_prune_num = 5
    pruning_prob = 0.5
    quantization_prob = 0.5
    distillation_prob = 0.5
    train_epoch_per_iter = 1
    lr = 2e-5
    possible_quantizations = ["16", "8", "4"]
    quality_treshold = 1
    device_count = torch.cuda.device_count() - 1

    models_fitness = []
    pick_pairing_probs = []
    models_score = []
    quantization_arr = [None] * pop_size
    models_size = []
    models_inference_time = []

    models_per_gpu = pop_size // device_count
    ost = pop_size % device_count
    mp.set_start_method("spawn")

    init_model_size = calc_model_size(model)
    init_model_score, init_model_time = calc_model_score_and_time(
        model, valid_dataloader, metric
    )
    best_model_score = init_model_score

    pick_pairing_probs = [1 / pop_size] * pop_size
    process_params = []
    for i in range(pop_size):
        model.save_pretrained(f"./test_model_{i}")
        gpu_number = (
            i % device_count
            if i >= device_count * models_per_gpu
            else i // models_per_gpu
        )
        process_params.append(
            [
                i,
                gpu_number,
                pop_size,
                a,
                b,
                c,
                max_prune_frac,
                max_attention_layers_prune_num,
                pruning_prob,
                quantization_prob,
                distillation_prob,
                train_epoch_per_iter,
                possible_quantizations,
                lr,
                train_dataloader,
                valid_dataloader,
                None,
                pick_pairing_probs,
            ]
        )
        models_fitness.append((init_model_score**a) / (init_model_size**b))
        models_score.append(init_model_score)
        models_size.append(init_model_size)
        models_inference_time.append(init_model_time)

    model.save_pretrained("./best_score_model")

    min_possible_model_score = init_model_score * quality_treshold

    pick_pairing_probs = np.array(pick_pairing_probs)
    best_model = None
    no_improve_epoch = 0
    counter = 0
    max_score = -1
    min_size = np.inf
    best_model_fitness = -1
    best_model_quant_method = None

    for i in tqdm.tqdm(range(max_epoch)):
        pool_cur = mp.Pool(max_proc)

        res_stats = pool_cur.starmap(process_model, process_params)
        models_fitness_new = list(
            map(lambda x: x[0] if x[1] > min_possible_model_score else 0, res_stats)
        )
        models_score_new = list(map(lambda x: x[1], res_stats))
        quantization_arr_new = list(map(lambda x: x[2], res_stats))
        models_size_new = list(map(lambda x: x[3], res_stats))
        models_inference_time_new = list(map(lambda x: x[4], res_stats))

        models_fitness = models_fitness + models_fitness_new
        models_score = models_score + models_score_new
        quantization_arr = quantization_arr + quantization_arr_new
        models_size = models_size + models_size_new
        models_inference_time = models_inference_time + models_inference_time_new

        print("len model_pop not filter: ", len(models_fitness))

        models_fitness_arr = np.array(models_fitness)

        models_fitness_arr = (
            models_fitness_arr / (init_model_score**a) * (init_model_size**b)
        )
        print(models_fitness_arr)

        if i == 0:
            remain_indexes = [0] + list(range(pop_size, pop_size * 2 - 1))
        else:
            order = np.argsort(models_fitness_arr)[::-1]
            remain_indexes = order[:pop_size].tolist()

        models_fitness_arr = models_fitness_arr[remain_indexes]

        print("len model_pop filter: ", len(models_fitness_arr))
        print("Fitnesses: ", models_fitness_arr)

        models_fitness = np.array(models_fitness)[remain_indexes].tolist()
        models_score = np.array(models_score)[remain_indexes].tolist()
        quantization_arr = np.array(quantization_arr)[remain_indexes].tolist()
        models_size = np.array(models_size)[remain_indexes].tolist()
        models_inference_time = np.array(models_inference_time)[remain_indexes].tolist()

        print("Scores: ", models_score)
        print("Sizes: ", models_size)
        print("Inference times: ", models_inference_time)
        print("Quantizations: ", quantization_arr)

        pick_pairing_probs = (
            np.exp(
                (np.array(models_score) - np.min(models_score))
                / (np.max(models_score) - np.min(models_score))
            )
            / np.sum(
                np.exp(
                    (np.array(models_score) - np.min(models_score))
                    / (np.max(models_score) - np.min(models_score))
                )
            )
        ).tolist()
        print("Piking probas: ", pick_pairing_probs)

        for i in range(pop_size):
            process_params[i][-1] = pick_pairing_probs
            if remain_indexes[i] != i:
                shutil.copytree(
                    f"./test_model_{remain_indexes[i]}",
                    f"./test_model_{i}",
                    dirs_exist_ok=True,
                )

        cur_best_model_score_ind = np.argmax(models_score)
        if models_score[cur_best_model_score_ind] > best_model_score:
            best_model_score = models_score[cur_best_model_score_ind]
            shutil.copytree(
                f"./test_model_{cur_best_model_score_ind}",
                "./best_score_model",
                dirs_exist_ok=True,
            )

        print("Best model score: ", best_model_score)

        pool_cur.close()
        pool_cur.terminate()
        pool_cur.join()
        del pool_cur
        gc.collect()
        torch.cuda.empty_cache()
