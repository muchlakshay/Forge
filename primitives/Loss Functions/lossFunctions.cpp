#include "lossFunctions.h"
#include "lossFunctionsAbstract.h"
#include "lossFunctionsGrads.h"
#include "../primitives/Dispatcher/primitive_dispatcher.h"
#include "autograd/attach_node.h"

void check_device_dtype_shape(const Forge::Tensor& pred, const Forge::Tensor& ground_truth) {
    if (pred.shape()!=ground_truth.shape()) throw std::invalid_argument(
        "predictions and labels have different shapes");
    if (pred.device() != ground_truth.device()) throw std::invalid_argument(
    "prediction and label tensor have different devices");
    if (pred.dtype() != ground_truth.dtype()) throw std::invalid_argument(
        "prediction and label tensor have different dtypes");
}

const Forge::Tensor& Forge::MSE::operator()(const Tensor &predictions, const Tensor &ground_truth) {
    check_device_dtype_shape(predictions, ground_truth);

    auto dispatch_key {predictions.dispatch_key()};
    auto device {predictions.device()};

    m_loss = Tensor{{1}, Dtype::float32, true, predictions.device()};

    const auto* mse {primitive_dispatcher().lookup<MSEAbstract>(primitive_ops::mse, dispatch_key)};
    mse->compute_loss(predictions, ground_truth, m_loss);

    attach_node<MSEGradsAbstract, 2>(m_loss, device, primitive_dispatcher(), primitive_ops::mse, predictions, ground_truth);
    return m_loss;
}

const Forge::Tensor& Forge::CrossEntropy::operator()(const Tensor &predictions, const Tensor &ground_truth) {
    check_device_dtype_shape(predictions, ground_truth);

    auto dispatch_key {predictions.dispatch_key()};
    auto device {predictions.device()};

    m_loss = Tensor{{1}, Dtype::float32, true, predictions.device()};

    const auto* ce {primitive_dispatcher().lookup<CrossEntropyAbstract>(primitive_ops::crossEntropy, dispatch_key)};
    attach_node<CrossEntropyGradsAbstract, 2>(m_loss, device, primitive_dispatcher(), primitive_ops::crossEntropy, predictions, ground_truth);
    ce->compute_loss(predictions, ground_truth, m_loss);

    return m_loss;
}

const Forge::Tensor& Forge::BinaryCrossEntropy::operator()(const Tensor &predictions, const Tensor &ground_truth) {
    check_device_dtype_shape(predictions, ground_truth);

    auto dispatch_key {predictions.dispatch_key()};
    auto device {predictions.device()};

    m_loss = Tensor{{1}, Dtype::float32, true, predictions.device()};

    const auto* bce {primitive_dispatcher().lookup<BinaryCrossEntropyAbstract>(primitive_ops::binaryCrossEntropy, dispatch_key)};
    attach_node<BinaryCrossEntropyGradsAbstract, 2>(m_loss, device, primitive_dispatcher(), primitive_ops::binaryCrossEntropy, predictions, ground_truth);
    bce->compute_loss(predictions, ground_truth, m_loss);

    return m_loss;
}