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

** đầu tiên xử lý data :(cái này lấy ở đâu ?)
ta nên lấy trong video là tốt nhất.(ban đầu mình search google tìm ảnh camera giao thông sau đó nó không tốt vì size ảnh trên google không đều train không tốt.
cách lấy ảnh trong link tham khảo có 1 đoạn code mình tách ra riêng thành 1 file cho bạn nào không rành
- link trong github luôn là file frame.py đó (cái này yêu cầu máy phải python ,nếu chạy trên google colab luôn thì phải đổi đường link input/output thành link trên google drive)
- nhớ là link trên window khác link trên google colab nhé.chỉnh sửa link cho đúng là ok)

** sau khi có các frame của các video rồi(1 tập ảnh trung bình 1s=10 ảnh ) chúng ta sẽ lựa các ảnh để label lại pretrain **

chúng ta sử dụng phần mềm labelImg.exe(trước search google có giờ không thấy) hoặc làm theo hướng dẫn https://github.com/tzutalin/labelImg

chúng ta có thể chia 3 class thành 2 model vì xe máy khá nhỏ và nhiều trong khi 3 class còn lại quá ít dẫn đến data train không đều.

chúng ta có thể lựa mỗi video tầm 150 ảnh .10 video là 1500 ảnh rồi.rất nhiều đó đối class 1 có thể giảm xuống 1 ít tầm 100 ảnh 1 video vì chỉ train riêng nó thôi không sợ
sau khi label xong chúng ta bỏ vào theo hướng dẫn trong bài

* API TF 2 pre train model *
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
- tải file generate_tfrecords.py  tại  https://github.com/sglvladi/TensorFlowObjectDetectionTutorial/tree/master/docs/source/scripts


```
TensorFlow
├───scripts
│   └───preprocessing
│     └───generate_tfrecord.py 
└───workspace
    └───training_demo
        ├───annotations
        │   └───label_map.pbtxt 
        ├───exported-models
        ├───images
        │   ├───test
        │   │     └───test images with corresponding XML files
        │   └───train
        │         └───train images with corresponding XML files
        ├───models
        │   └───my_ssd_resnet50_v1_fpn
        │     └───pipeline.config
        └───pre-trained-models
            └───ssd_resnet152_v1_fpn_640x640_coco17_tpu-8
```

