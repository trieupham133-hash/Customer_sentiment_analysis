# A. Vấn đề và một cách giải quyết của nhóm 
### Vấn đề:
* Trong kinh doanh, bán hàng online trong hiện tại đã và đang bước vào thời kỳ hoàng kim, số doanh nghiệp, số lượng đơn hàng trực tuyến tăng vọt. Vì vậy, các vấn đề về việc phân tích và cải thiện doanh số của các doanh nghiệp cũng theo đó mà tăng vọt, và một trong đó là **phân tích các đánh giá của khách mua hàng**. Đây là một vấn đề khó khăn một khi số đánh giá tăng lên quá nhiều, ta không thể nào đọc từng dòng hoặc có thể thuê người đọc, đánh giá và lọc, nhưng sẽ vô cùng mất thời gian và tốn kém. Vậy giải quyết vấn đề như thế nào bây giờ? talk to me babe?

### Một lời giải của nhóm: 
* Trong thời đại mà đến cảm xúc còn có thể được cài đặt cho máy móc thì việc ta không dùng đến chúng là một sự lỗi thời và xuống cấp mạnh của hiệu suất công việc. Vì vậy theo nhóm, một giải pháp tân thời, ta có thể sử dụng **mô hình học sâu** để phân tích và phân loại tính chất của nhận xét. 
* Giới thiệu sơ bộ về mô hình mà nhóm dùng: Long Short-Term Memory (hay LSTM) là một mô hình học sâu được cải tiến dựa trên RNNs (Recurrent Neural Networks) với ý tưởng đại khái là máy móc sẽ "đọc" từng chữ, "nhớ", "đánh giá" rồi "phân loại"(Ý tưởng sẽ được giải thích rõ hơn ở phần sau).


--- 

# B. Mô tả chi tiết

## Phần 1: Dữ liệu
### Nguồn:
Dữ liệu được chia sẻ trên trang Kaggle. Link: https://www.kaggle.com/datasets/dduongdev/shopee-vietnamese-product-reviews-sentiment

### Mô tả dữ liệu: 
Dữ liệu gồm hơn 10000 dòng dữ liệu được chia vào thành 2 tập shoppee và augmented về đánh giá sản phẩm trên shoppee của các doanh nghiệp bán lẻ, gồm 4 cột:
```text
id  |  review   |   rating  |   label
```
Ý nghĩa của các cột:
- **id**: Khóa chính, là mã số của đánh giá.
- **review**: Chứa đánh giá của các khách hàng.
- **rating**: Số sao mà khách hàng đánh giá cho sản phẩm.
- **label**: Nhãn thật sự của đánh giá.  

Theo các phân tích, bộ dữ liệu khá sạch khi về căn bản không có dòng nào chứa null, không có dòng bị trùng lặp, các nhãn được ghi đầy đủ.
- Review dài nhất chứa 2021 ký tự, ngắn nhất 20 ký tự.
- Trong bộ dữ liệu gốc cũng chứa các ký tự lạ, emoji/icon, có chứa từ sai chính tả, có từ bị kéo dài  

### EDA:
#### 1/ Trực quan hóa không gian đặc trưng bằng t-SNE (Feature Separability Analysis)

<div align="center" style="font-weight: bold"> t-SNE Projection of Review TF-IDF Features (Shopee Dataset) </div>

![alt text](figures/tsne_feature_separability.png)
**Nhận xét:** Biểu đồ cho thấy **hai nhóm cảm xúc** (`positive` và `negative`) có **sự chồng lấn (overlap) tương đối lớn**, nhưng vẫn **bộc lộ xu hướng phân cụm (clustering trend) theo vùng rõ rệt**.
- Các đặc trưng TF-IDF thuần túy (tần suất từ) ở mức cơ bản đủ để thuật toán nhận diện được sự khác biệt giữa hai sắc thái, nhưng chưa đủ mạnh để phân tách chúng thành hai cụm hoàn toàn biệt lập.

<div align="center" style="font-weight: bold"> t-SNE Projection of Review TF-IDF Features (Augmented Dataset) </div>

![alt text](figures/tsne_feature_separability_aug.png)
**Nhận xét:** Biểu đồ cho thấy **khả năng phân tách tuyến tính (separability) giữa hai nhãn cảm xúc đã được cải thiện rõ rệt** so với tập dữ liệu Shopee gốc.
- Thay vì trộn lẫn phức tạp ở khu vực trung tâm, hai cụm dữ liệu giờ đây đã phân hóa theo trục dọc (`tsne_2`). Nhãn `positive` (màu xanh) chiếm lĩnh hoàn toàn nửa trên của không gian biểu đồ, trong khi nhãn `negative` (màu đỏ) tập trung chủ yếu ở nửa dưới.

#### 2/ Phân phối số lượng mẫu và nhãn
<div align="center" style="font-weight: bold"> Biểu đồ phân phối nhãn </div>

![alt text](figures/class_distribution.png)

**Nhận xét:** Biểu đồ trên cho thấy 
- **Tập dữ liệu gốc (Shopee Dataset)**  
  - **Phân bố**: Nhãn `negative` chiếm ưu thế áp đảo với **5,965 mẫu**, trong khi nhãn `positive` chỉ có **3,634 mẫu**.
  - **Đặc trưng:** Nhóm tiêu cực chiếm khoảng $62%$ tổng số dữ liệu. Hơi là lạ, doanh nghiệp này có lẽ cần phải kiểm tra lại phương thức bán hàng. Nhưng với vấn đề học máy, đây chỉ là một sự lệch dữ liệu nhỏ, không đáng kể, không ảnh hưởng quá sâu sắc đến khả lăng học, chỉ hơi bias nhẹ một tí cho nhãn `negative` thôi.

- **Tập dữ liệu tăng cường (Augmented Dataset)**
  - **Phân bố:** Nhãn`negative` có **677 mẫu** và `positive` có **671 mẫu**.
  - **Đặc trưng:** Việc chủ động đưa tỷ lệ về mức xấp xỉ $50:50$ giúp loại bỏ hoàn toàn yếu tố thiên vị class khi huấn luyện.  
  <br>


<div align="center" style="font-weight: bold"> Biểu đồ phân phối rating của khách hàng </div>

![alt text](figures/rating_distribution.png)
**Nhận xét:** Nhìn chung, cấu hình phân bổ điểm đánh giá giữa hai tập dữ liệu giữ **nguyên hình dáng tổng thể** (distribution shape). Cả hai đều mang đặc trưng của dữ liệu đánh giá thương mại điện tử thực tế: **Phân hóa mạnh về hai đầu cực đoan** (tập trung rất nhiều vào `1 sao` và `5 sao`, cực kỳ ít ở `4 sao`).

Điều này chứng tỏ quá trình lấy mẫu (sampling) hoặc trích xuất để tạo tập `Augmented Dataset` đã **bảo toàn khá tốt tỷ lệ cấu trúc rating gốc**, không làm lệch phân phối điểm số ban đầu của người dùng.

#### 3/ Mối quan hệ giữa điểm đánh giá và nhãn cảm xúc
<div align="center" style="font-weight: bold"> Biểu đồ cột thể hiện mối quan hệ giữa nhãn và đánh giá </div>

![alt text](figures/label_by_rating_distribution.png)

**Nhận xét:** Biểu đồ này thể hiện một sự thật quan trọng: **Tập dữ liệu gốc (Shopee) bị nhiễu nhãn (Label Noise) rất nặng**, đặc biệt ở các dải điểm trung gian như `3 sao` và `4 sao` ***(Cần thêm lý do vào đây)***. Khi sang tập **Augmented Dataset**, mặc dù cấu trúc phân phối rating được giữ nguyên, nhưng mối quan hệ gán nhãn đã được phân định rạch ròi và "sạch" hơn đáng kể.

Do đồ thị dùng trục tung dạng **Log Scale**, các thanh bar thấp nhìn có vẻ không chênh lệch nhiều, nhưng thực tế số lượng chênh lệch tuyến tính là cực kỳ lớn.

#### 4/ Kết luận chung sau EDA
<div align="center" style="font-weight: bold"> Tổng quan cấu trúc hệ thống dữ liệu </div>

| Đặc trưng phân tích | Tập dữ liệu gốc (Shopee Dataset) | Tập dữ liệu tăng cường (Augmented Dataset) |
| :--- | :--- | :--- |
| **Số lượng mẫu** | 9,599 mẫu (Khá lớn) | 1,348 mẫu (Đã qua lọc/Downsampling) |
| **Cân bằng nhãn** | Mất cân bằng nghiêm trọng (62% Negative : 38% Positive) | **Cân bằng lý tưởng (50% Negative : 50% Positive)** |
| **Quy tắc gán nhãn** | Bị nhiễu nặng (Label Noise) ở mức 3 và 4 sao | Được làm sạch và chuẩn hóa triệt để theo mức Rating |
| **Độ dài văn bản** | Phân phối tự nhiên (Negative dài hơn Positive một chút) | Nhân tạo (Negative bị kéo dài bất thường, tối đa ~155 từ) |
| **Không gian ngữ nghĩa** | Chồng lấn mạnh ở trung tâm (Từ vựng trung tính nhiều) | **Phân tách tuyến tính rõ rệt trên trục dọc t-SNE** |

**Caution:** Bộ dữ liệu Augmented quá "đẹp", quá "hoàn hảo", nhưng ông cha ta nói, cái gì quá cũng không tốt. Trong trường hợp này sự đẹp đó sẽ khiến mô hình sẽ học vẹt, và khi đưa  mô hình này deploy dự đoán các đánh giá tự nhiên ngoài đời thực, độ chính xác giảm là điều mà ta không thể tránh (Drop Performance).

### Preprocessing
Trong phần này, ta xem như đã gộp 2 dataset trên thành 1 để dễ hoạt động, tuy nhiên đối với mỗi mô hình, yêu cầu dữ liệu đầu vào là khác nhau, nên nhóm chia thành hai nhánh xử lí:
- Một nhánh cho baseline model, tức là những model đơn giản như SVM, -> Cần xử lí dữ liệu đầu vào phức tạp hơn, kỹ hơn một tí
- Một nhánh cho neural network model, như LSTM hay PhoBERT -> Không cần xử lí quá kỹ, để giữ được nhiều thông tin hơn.
</br>

#### 1/ Clean text
Trong phần này, trước hết, cần xử lí chung cho cả hai mô hình những điều như sau:
- Chuẩn hóa khoảng trắng: loại bỏ các khoảng trắng dư thừa trong các câu.
- Chuyển emoji thành dạng chữ: ở đây chúng em xử lí đơn giản bằng cách mapping các emoji thông dụng thành các dạng `positive_emoji`, `negative_emoji`, ...
- Loại bỏ URL, HTML, email được dán kèm trong review.

Riêng đối với mô hình baseline, ta cần xử lí khắc khe hơn một vài điều sau:
- Chuyển chữ hoa thành chữ thường
- Loại bỏ dấu câu

#### 2/ Normalize text
Một trong những quy trình khó khăn nhất trong preprocessing, khi hiện tại vẫn có khá ít thư viện có thể xử lí một cách triệt để tiếng Việt. Một số điều nhóm em đã xử lí như sau:
- **Chuẩn hóa lại các từ bị kéo dài:** ví dụ như "Phongggggg đẹpp trai" -> "Phongg đẹpp trai". Nhóm áp dụng kiểu chuyển nếu chứa từ 3 chữ lặp lại trở lên thì mới giảm xuống còn hai vì hai lý do chính: 
    - Tránh loại bỏ mất các từ tiếng anh như free, good,... (lí do chính)
    - Giữ lại một tí thông tin, khi các từ kéo dài thường biểu thị cảm xúc "mạnh" hơn
<br>
- **Chuẩn hóa các từ viết tắt/slang:** Ở đây bọn em cũng chỉ dùng một cách đơn giản là mapping lại một số từ thông dụng rồi chuyển đổi,  
