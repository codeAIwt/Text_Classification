import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import random
import time
import os

# 设置随机种子以确保结果可重现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用tokenizer进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data_from_txt(folder_path):
    """从txt文件加载数据集"""
    texts = []
    labels = []
    
    # 获取文件夹中所有txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 从文件名提取标签（假设文件名是数字编号，如0.txt, 1.txt等）
                label = int(filename.split('.')[0])
                
                texts.append(text)
                labels.append(label)
                
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
                continue
    
    if not texts:
        raise ValueError("没有成功加载任何文本文件")
    
    # 转换为numpy数组
    texts = np.array(texts)
    labels = np.array(labels)
    
    print(f"共加载 {len(texts)} 个文本文件")
    print(f"标签类别: {np.unique(labels)}")
    print(f"每个类别的样本数: {np.bincount(labels)}")
    
    return texts, labels

def create_dataloaders(train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size=16):
    """创建数据加载器"""
    train_dataset = TextClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer
    )
    
    test_dataset = TextClassificationDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, test_dataloader

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """训练一个epoch"""
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """评估模型"""
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, scheduler, num_epochs=3, model_save_path='./model/best_model_state.bin'):
    """完整的模型训练流程"""
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 10)
        
        train_acc, train_loss = train_epoch(
            model, train_dataloader, loss_fn, optimizer, device, scheduler, len(train_dataloader.dataset)
        )
        
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        
        test_acc, test_loss = eval_model(
            model, test_dataloader, loss_fn, device, len(test_dataloader.dataset)
        )
        
        print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
        print()
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            # 保存模型
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
    
    # 训练结束后强制保存最终模型
    final_model_path = os.path.splitext(model_save_path)[0] + '.bin'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    return best_accuracy

def get_predictions(model, data_loader, device):
    """获取模型预测结果"""
    model = model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    return true_labels, predictions

def main():
    """主函数"""
    # 设置参数
    RANDOM_SEED = 42
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_SAVE_PATH = './model/best_model_state.bin'  # 模型保存路径
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 设置设备
    device = setup_device()
    
    # 1. 加载数据
    print("加载数据...")
    local_bert_path = "./bert-base-chinese"  # 本地BERT模型路径
    data_dir = "./邮件_files"  # 邮件数据文件夹路径
    
    # 加载邮件文本和标签
    texts, labels = load_data_from_txt(data_dir)
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(local_bert_path)
    
    # 2. 划分训练集和测试集
    print("划分训练集和测试集...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # 3. 创建数据加载器
    print("创建数据加载器...")
    train_dataloader, test_dataloader = create_dataloaders(
        train_texts, train_labels, test_texts, test_labels, tokenizer, batch_size=BATCH_SIZE
    )
    
    # 4. 加载预训练的BERT模型
    print("加载BERT模型...")
    num_classes = len(np.unique(np.concatenate([train_labels, test_labels])))
    model = BertForSequenceClassification.from_pretrained(
        local_bert_path,
        num_labels=num_classes
    )
    model = model.to(device)
    
    # 检查是否存在已保存的模型
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"找到已保存的模型 {MODEL_SAVE_PATH}，正在加载...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("模型加载完成！")
        
        # 直接进行评估
        print("\n评估模型性能...")
        true_labels, predictions = get_predictions(model, test_dataloader, device)
        
        print("\n分类报告:")
        print(classification_report(true_labels, predictions))
    else:
        print("未找到已保存的模型，开始训练...")
        
        # 5. 设置优化器和学习率调度器
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_dataloader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 6. 定义损失函数
        loss_fn = nn.CrossEntropyLoss().to(device)
        
        # 7. 训练模型
        print("开始训练模型...")
        start_time = time.time()
        
        best_accuracy = train_model(
            model, train_dataloader, test_dataloader, loss_fn, optimizer, device, scheduler, 
            NUM_EPOCHS, MODEL_SAVE_PATH  # 传递模型保存路径
        )
        
        end_time = time.time()
        print(f"训练完成！耗时: {end_time - start_time:.2f}秒")
        print(f"最佳测试准确率: {best_accuracy:.4f}")
        
        # 8. 评估模型
        print("\n评估模型性能...")
        true_labels, predictions = get_predictions(model, test_dataloader, device)
        
        print("\n分类报告:")
        print(classification_report(true_labels, predictions))

if __name__ == "__main__":
    main()