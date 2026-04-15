/* change paths if you want to train it yourself!!
 * change path where to download MNIST dataset and then of the train/test files */

#include "../../data/Extra/getMNIST.h"
#include "load_mnist.h"
#include "../../Forge.h"

struct MLP {
    Forge::Linear layer_1{784, 16};
    Forge::Linear layer_2{16, 16};
    Forge::Linear layer_3{16, 10};
    Forge::Relu relu;
    auto operator()(const Forge::Tensor& input) const {return layer_3(relu(layer_2(relu(layer_1(input)))));}
    auto parameters() {
        Forge::Parameter l1_w {&layer_1.weights(), true};
        Forge::Parameter l2_w {&layer_2.weights(), true};
        Forge::Parameter l3_w {&layer_3.weights(), true};
        Forge::Parameter l1_b {&layer_1.bias()};
        Forge::Parameter l2_b {&layer_2.bias()};
        Forge::Parameter l3_b {&layer_3.bias()};

        std::vector params {l1_w, l2_w, l3_w, l1_b, l2_b, l3_b};
        return params;
    }
};

void downloadMNIST() {
    static bool downloaded {};
    if (!downloaded) {
        Forge::Extra::getMNIST("Data/"); //change path accordingly
        downloaded = true;
    }
}

int main() {
    downloadMNIST();
    //change these paths according to where you downloaded MNIST
    auto train_img_path {"Data/train-images.idx3-ubyte"};
    auto train_lab_path {"Data/train-labels.idx1-ubyte"};
    auto test_img_path {"Data/t10k-images.idx3-ubyte"};
    auto test_lab_path {"Data/t10k-labels.idx1-ubyte"};

    auto [images, labels] = load_mnist(train_img_path, train_lab_path);
    auto [test_images, test_labels] = load_mnist(test_img_path, test_lab_path);

    images = images.reshape(1875, 32, 784);
    labels = labels.reshape(1875, 32, 10);
    test_images = test_images.reshape(10000, 784);
    test_labels = test_labels.reshape(10000, 10);

    MLP mlp {};
    Forge::CrossEntropy CE;
    Forge::Adam optimizer{mlp.parameters(), 0.001f};
    optimizer.setDecayFactor(0.02f);

    int epochs {1};
    for (int epoch{1}; epoch <= epochs; ++epoch) {
        float batch_loss {};
        for (int batch{}; batch<images.shape()[0]; ++batch) {
            auto opt {mlp(images[batch])};
            auto loss {CE(opt, labels[batch])};
            loss.backward();
            optimizer.update();
            optimizer.clear_grads();
            batch_loss += *static_cast<float*>(loss.data());
        }
        std::cout<<"Epoch: "<<epoch<<" | Loss: "<<batch_loss/images.shape()[0]<<"\n";
    }
    Forge::Softmax softmax;
    auto predictions {softmax(mlp(test_images))};
    for (int i {}; i<test_labels.shape()[0]; ++i) {
        std::cout<<"Ground Truth: \n"<<test_labels[i]<<"\nPredicted: \n"<<predictions[i]<<"\n\n";
    }

    /*
     * STATS AFTER TRAINING FOR 100 EPOCHS USING DIFFERENT OPTIMIZERS
     * Adam:  Loss: 0.469373 -> 0.0438026 (lr=0.001, beta_1=0.9, beta_2=0.999)
     * AdamW: Loss: 0.464487 -> 0.0489543 (lr=0.01,  beta_1=0.9, beta_2=0.999, decay_rate=0.02)
     * SGD:   Loss: 2.17553  -> 0.192936  (lr=0.001)
     * SGD:   Loss: 0.980079 -> 0.0790806 (lr=0.01 )
     * SGD with momentum: Loss: 0.852001 -> 0.077168 (lr=0.01, momentum_coef=0.2)
     */

    return 0;
}
