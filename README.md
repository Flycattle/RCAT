This repository is for “RESIDUAL CLASS ATTENTION TRANSFER : SIMPLIFYING KNOWLEDGE DISTILLATION FROM VISION FOUNDATION MODELS TO CNNS”
<img width="2489" height="961" alt="image" src="https://github.com/user-attachments/assets/53dcfdde-ca3a-4122-a376-eedc83bad396" />

image
The training log can be viewed in the /output/train directory. 训练日志可以查看，在/output/train 下 文件夹名包含rcat,vca都为RCAT方法的训练日志
 For how to reproduce, please refer to the file get_start. txt


如何复现，请参考文件 get_start.txt


下面涉及到路径的地方，需要更换为自己的路径，例如"train.py"(替换为你的 train.py 路径) "--config xxxx"（xxx 替换为你的配置文件路径） "--teacher-pretrained xxxx" （xxxx 替换为你的教师模型文件路径）"/data/cifar100"（替换为你的数据集所在路径）这四项换成你的路径

To reproduce, refer to the file get_start.txt
For path-related sections, replace the following with your own paths: "train.py" (replace with your train.py path), "--config xxxx" (replace xxxx with your configuration file path), "--teacher-pretrained xxxx" (replace xxxx with your teacher model file path), and "/data/cifar100" (replace with your dataset path).

For example, use the command for our method-RCAT:


例如，使用该命令复现我们的方法-RCAT:


CUDA_VISIBLE_DEVICES=0 python /nfs4/wangyb/projects/RCAT/train.py
/nfs4/wangyb/projects/OFAKD/data/cifar100 --config /nfs4/wangyb/projects/RCAT/configs/cifar/cnn.yaml
--model resnet18_proj_dim_384 --teacher dinov2s_finetuehead --teacher-pretrained /nfs4/wangyb/projects/OFAKD/teacher_checkpoint/dinov2_cifar100_head_best.pth
-d cifar100 --num-classes 100 --distiller rcat


use the command for other baseline methods:


使用该指令对其他基线方法：
Use this command for other baseline methods:

CUDA_VISIBLE_DEVICES=0 python train.py
/data/cifar100 --config configs/cifar/cnn.yaml
--model resnet18 --teacher dinov2s_finetuehead --teacher-pretrained teacher_checkpoint/dinov2_cifar100_head_best.pth
-d cifar100 --num-classes 100 --distiller kd


注意：train.py --config --teacher-pretrained /data/cifar100 这四项换成你的路径

Efficiency：
SwinT-to-Mobilenetv2 (Tiny-imagenet):
<img width="1147" height="997" alt="image" src="https://github.com/user-attachments/assets/46a231bd-6775-4d13-a3cd-1a3150016cec" />
Dinov2s-to-ResNet18(Tiny-ImageNet):
<img width="643" height="218" alt="image" src="https://github.com/user-attachments/assets/e6fd1319-757a-4c72-acb2-969364ce3b5a" />



消融：(ablation)

ρ： CIFAR100: dinov2s-to-ResNet18  
<img width="305" height="160" alt="image" src="https://github.com/user-attachments/assets/e04cde7a-638f-459c-a3c2-53f77eba0f51" />

ρ (超参数)	准确率 (Accuracy)
0.4	  83.15%
0.6	  84.02%
0.8	  84.18% (Best)
1	    83.65%
β和γ在表一中做了两个模块的分析，更多的消融结果如下：
β and γ conducted analyses of two modules in Table 1, with additional ablation results as follows:
<img width="880" height="272" alt="image" src="https://github.com/user-attachments/assets/8e2c1410-38bc-4412-a0c2-7f6d537c0ea6" />






