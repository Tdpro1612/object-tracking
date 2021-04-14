## đếm số xe qua camera giao thông trong 1 khoảng thời gian

nguồn data 
video : https://drive.google.com/drive/folders/1LHZFrl72IrDrXxudG9Mss6RG-KKnF4WI?usp=sharing
frame :https://drive.google.com/drive/folders/1Qaya1m_r8m869ZQMOQ-WVB4-wJOSdz-x?usp=sharing  (link bạn mình up,khả năng sẽ có khi die.nên mình sẽ hướng dẫn cách lấy luôn)

hướng dẫn cách làm
có 3 bước
- Phát hiện phương tiện giao thông trong từng frame của video: Sử dụng TensorFlow Object Detection API. Kết quả trả về là một danh sách các bounding box ứng với tất cả các phương tiện giao thông trong ảnh.
- Theo vết (multiple objects tracking): Dựa vào IOU (chỉ số đo đạc mức độ trùng lắp của hai bounding box), các bounding box ở các frame liên tiếp sẽ được gom nhóm và từ đó sẽ hình thành quỹ đạo di chuyển của chính phương tiện đó.
- Xác định hướng di chuyển (MOI) dựa trên quỹ đạo: MOI của phương tiện sẽ được lựa chọn dựa trên độ tương đồng (ở đây sử dụng Cosine Similarity Score) giữa quỹ đạo của mỗi phương tiện và các MOI sẽ được tính toán.

link tham khảo : https://github.com/hcmcaic/ai-challenge-2020

## Bài giải
nhận xét bài toán
đầu tiên bạn nên chạy hết file của link tham khảo để xem xét các bước nó làm thế nào và kết quả nó ra sao để có thể hiểu được bài toán giải thế nào.
lưu ý dòng code phần Đọc dữ liệu từ video
``` 
from tqdm import tqd
sửa thành
from tqdm import tqdm #@markdown Your videos is stored in:
```

nó bị lỗi thiếu chữ m thôi nhé sửa lại là ok run thôi 

qua bài tham khảo ta có thể thấy model train sẵn không tốt lắm chúng ta cần train lại theo các class ta cần 
- class 1 : xe 2 bánh : xe máy,xe đạp
- class 2 : ô tô con 4-6 chỗ ngồi
- class 3 : ô tô 9 chỗ trở lên,xe buýt
- class 4 : xe tải,xe container

### bước 1 : pretrain lại model
*** đối với sinh viên có thể sử dụng google colab để train tạm(vì nghèo làm gì có máy GPU :D)
ở đây hướng dẫn bằng cách xài Tensorflow 2 API để nhận diện vật thể(object detection)

**đầu tiên xử lý data :(cái này lấy ở đâu ?)**
ta nên lấy trong video là tốt nhất.(ban đầu mình search google tìm ảnh camera giao thông sau đó nó không tốt vì size ảnh trên google không đều train không tốt.
cách lấy ảnh trong link tham khảo có 1 đoạn code mình tách ra riêng thành 1 file cho bạn nào không rành
- link trong github luôn là file frame.py đó (cái này yêu cầu máy phải python ,nếu chạy trên google colab luôn thì phải đổi đường link input/output thành link trên google drive)
- nhớ là link trên window khác link trên google colab nhé.chỉnh sửa link cho đúng là ok)

**sau khi có các frame của các video rồi(1 tập ảnh trung bình 1s=10 ảnh ) chúng ta sẽ lựa các ảnh để label lại pretrain**

- chúng ta sử dụng phần mềm labelImg.exe(trước search google có giờ không thấy) hoặc làm theo hướng dẫn https://github.com/tzutalin/labelImg

- chúng ta có thể chia 4 class thành 2 model vì xe máy khá nhỏ và nhiều trong khi 3 class còn lại quá ít dẫn đến data train không đều.

- chúng ta có thể lựa mỗi video tầm 150 ảnh .10 video là 1500 ảnh rồi.rất nhiều đó đối class 1 có thể giảm xuống 1 ít tầm 100 ảnh 1 video vì chỉ train riêng nó thôi không sợ
sau khi label xong chúng ta bỏ vào theo hướng dẫn trong bài
sau khi label xong chúng ta chia thành 2 forder train/test và file xml kèm theo
-tải và extract model chúng ta chọn tại https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md  
Ở đây sử dụng ssd_resnet152_v1_fpn_640x640_coco17_tpu-8
*API TF 2 pre train model*
- tạo 1 cây thư mục giống vậy

```
TensorFlow
├───scripts
│   └───preprocessing
└───workspace
    └───training_demo
        ├───annotations
        ├───exported-models
        ├───images
        │   ├───test
        │   └───train
        ├───models
        └───pre-trained-models
 ```       
- tạo file label_map.pbxt
```
item {
      id: 1
      name: 'class 1'
}
name chính là tên class mà khi ta label ta lấy ví dụ class 1  khi ta label là class1 thì chỗ này phải để là class1 không được có dấu cách,bao nhiêu class thì để bấy nhiêu item thôi
```
- chỉnh sửa file pipeline.config 
```
Line 3:
num_classes: 1 (#number of classes your model can classify/ number of different labels)
Line 131:
batch_size: 1 (#you can read more about batch_size here)
Line 161:
fine_tune_checkpoint: "pre-trained-models/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" (#path to checkpoint of downloaded pre-trained-model)
Line 162:
num_steps: 50000 (#maximum number of steps to train model,max is 250000 note that this specifies the maximum number of steps, you can stop model training on any step you wish)
Line 167:
fine_tune_checkpoint_type: "detection" (#since we are training full detection model, you can read more about model fine-tuning here)
Line 168:
use_bfloat16: false (#Set this to true only if you are training on a TPU)
Line 172:
label_map_path: "annotations/label_map.pbtxt" (#path to your label_map file)
Line 174:
input_path: "annotations/train.record" (#path to train.record)
Line 182:
label_map_path: "annotations/label_map.pbtxt" (#path to your label_map file)
Line 186:
input_path: "annotations/test.record" (#Path to test.record)
```
- sau đó bỏ các file vào trong các thư mục đã tạo theo sơ đồ sau
```
TensorFlow
├───scripts
│   └───preprocessing
│     └───**generate_tfrecord.py** 
└───workspace
    └───training_demo
        ├───annotations
        │   └───**label_map.pbtxt**
        ├───exported-models
        ├───images
        │   ├───test
        │   │     └───**test images with corresponding XML files**
        │   └───train
        │         └───**train images with corresponding XML files**
        ├───models
        │   └───my_ssd_resnet50_v1_fpn
        │     └───**pipeline.config**
        └───pre-trained-models
            └───**ssd_resnet152_v1_fpn_640x640_coco17_tpu-8**
**model_main_tf2.py**
**exporter_main_v2.py**
```

Sau khi sắp xêp ảnh theo sơ đồ trên
vào file **API TF 2 pre train model.ipynb** run thôi

sau khi chạy xong ta sẽ có các model đã pretrain lại các class,detect 1 vài sample trong đó

### Bước 2 & 3 phát hiện MOI và ROI thôi nào
run file track class 1.ipynb  để truy xuất kết quả .(cái này viết lại từ file dl_detect_tf2_2_3.ipynb)

nó sẽ track từng frame như thế này và sau đó sẽ gộp trong IOU

![track frame](https://user-images.githubusercontent.com/61773507/100399181-ff8c6b00-300e-11eb-8bf6-d7efc1554839.jpg)

trong quá trình run test các frame ta có thể thấy rằng 

![test](https://user-images.githubusercontent.com/61773507/100399111-b6d4b200-300e-11eb-8d52-f16d1995e0a0.jpg)

thay đổi min sc có thể hữu hiệu track các vật phẩm nên ta có thể lựa chon 1 con số sc sao cho phù hợp


**ĐÂY LÀ KẾT QUẢ TEST CLASS 1  có dạng thế này**

![test class1](https://user-images.githubusercontent.com/61773507/100400407-b1c63180-3013-11eb-8819-9ebf5c7d30c9.jpg)


** TƯƠNG TỰ VỚI CÁC CLASS CÒN LẠI THÔI
GOOD LUCK
