import torch
import bitsandbytes as bnb

# 检查 PyTorch 是否可用 CUDA
print("PyTorch CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 创建一个简单的模型
model = torch.nn.Sequential(
    torch.nn.Linear(10000, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10)
).cuda()

# 使用 bitsandbytes 的 8-bit Adam 优化器
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)

# 创建随机输入和标签
x = torch.randn(64, 10000, device="cuda")
y = torch.randint(0, 10, (64,), device="cuda")

criterion = torch.nn.CrossEntropyLoss()

# 运行几步训练
for step in range(3):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Step {step+1}, Loss: {loss.item():.4f}")

print("✅ bitsandbytes 测试完成，GPU 已参与计算。")
