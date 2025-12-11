This repository is for “RESIDUAL CLASS ATTENTION TRANSFER : SIMPLIFYING KNOWLEDGE DISTILLATION FROM VISION FOUNDATION MODELS TO CNNS”
<img width="2489" height="961" alt="image" src="https://github.com/user-attachments/assets/53dcfdde-ca3a-4122-a376-eedc83bad396" />

image
The training log can be viewed in the /output/train directory. 训练日志可以查看，在/output/train下 For how to reproduce, please refer to the file get_start. txt

如何复现，请参考文件get_start.txt

下面涉及到路径的地方，需要更换为自己的路径，例如"train.py"(替换为你的train.py路径) "--config xxxx"（xxx替换为你的配置文件路径） "--teacher-pretrained xxxx" （xxxx替换为你的教师模型文件路径）"/data/cifar100"（替换为你的数据集所在路径）这四项换成你的路径

For example, use the command for our method-RCAT:

例如，使用该命令复现我们的方法-RCAT:

CUDA_VISIBLE_DEVICES=0 python /nfs4/wangyb/projects/RCAT/train.py
/nfs4/wangyb/projects/OFAKD/data/cifar100 --config /nfs4/wangyb/projects/RCAT/configs/cifar/cnn.yaml
--model resnet18_proj_dim_384 --teacher dinov2s_finetuehead --teacher-pretrained /nfs4/wangyb/projects/OFAKD/teacher_checkpoint/dinov2_cifar100_head_best.pth
-d cifar100 --num-classes 100 --distiller rcat

use the command for other baseline methods:

使用该指令对其他基线方法：

CUDA_VISIBLE_DEVICES=0 python train.py
/data/cifar100 --config configs/cifar/cnn.yaml
--model resnet18 --teacher dinov2s_finetuehead --teacher-pretrained teacher_checkpoint/dinov2_cifar100_head_best.pth
-d cifar100 --num-classes 100 --distiller kd

注意：train.py --config --teacher-pretrained /data/cifar100这四项换成你的路径
