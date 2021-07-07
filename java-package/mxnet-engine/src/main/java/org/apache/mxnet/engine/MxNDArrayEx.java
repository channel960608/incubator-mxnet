package org.apache.mxnet.engine;

import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.ndarray.NDList;
import org.apache.mxnet.api.ndarray.index.NDArrayIndexer;
import org.apache.mxnet.api.ndarray.internal.NDArrayEx;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.api.ndarray.types.SparseFormat;

import java.util.List;

public class MxNDArrayEx implements NDArrayEx {
    @Override
    public NDArray rdiv(Number n) {
        return null;
    }

    @Override
    public NDArray rdiv(NDArray b) {
        return null;
    }

    @Override
    public NDArray rdivi(Number n) {
        return null;
    }

    @Override
    public NDArray rdivi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rsub(Number n) {
        return null;
    }

    @Override
    public NDArray rsub(NDArray b) {
        return null;
    }

    @Override
    public NDArray rsubi(Number n) {
        return null;
    }

    @Override
    public NDArray rsubi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rmod(Number n) {
        return null;
    }

    @Override
    public NDArray rmod(NDArray b) {
        return null;
    }

    @Override
    public NDArray rmodi(Number n) {
        return null;
    }

    @Override
    public NDArray rmodi(NDArray b) {
        return null;
    }

    @Override
    public NDArray rpow(Number n) {
        return null;
    }

    @Override
    public NDArray rpowi(Number n) {
        return null;
    }

    @Override
    public NDArray relu() {
        return null;
    }

    @Override
    public NDArray sigmoid() {
        return null;
    }

    @Override
    public NDArray tanh() {
        return null;
    }

    @Override
    public NDArray softPlus() {
        return null;
    }

    @Override
    public NDArray softSign() {
        return null;
    }

    @Override
    public NDArray leakyRelu(float alpha) {
        return null;
    }

    @Override
    public NDArray elu(float alpha) {
        return null;
    }

    @Override
    public NDArray selu() {
        return null;
    }

    @Override
    public NDArray gelu() {
        return null;
    }

    @Override
    public NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return null;
    }

    @Override
    public NDArray globalMaxPool() {
        return null;
    }

    @Override
    public NDArray avgPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode, boolean countIncludePad) {
        return null;
    }

    @Override
    public NDArray globalAvgPool() {
        return null;
    }

    @Override
    public NDArray lpPool(float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return null;
    }

    @Override
    public NDArray globalLpPool(float normType) {
        return null;
    }

    @Override
    public void adadeltaUpdate(NDList inputs, NDList weights, float weightDecay, float rescaleGrad, float clipGrad, float rho, float epsilon) {

    }

    @Override
    public void adagradUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float epsilon) {

    }

    @Override
    public void adamUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float beta1, float beta2, float epsilon, boolean lazyUpdate) {

    }

    @Override
    public void nagUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float momentum) {

    }

    @Override
    public void rmspropUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float rho, float momentum, float epsilon, boolean centered) {

    }

    @Override
    public void sgdUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float momentum, boolean lazyUpdate) {

    }

    @Override
    public NDList convolution(NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding, Shape dilation, int groups) {
        return null;
    }

    @Override
    public NDList deconvolution(NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding, Shape outPadding, Shape dilation, int groups) {
        return null;
    }

    @Override
    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        return null;
    }

    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
        return null;
    }

    @Override
    public NDList prelu(NDArray input, NDArray alpha) {
        return null;
    }

    @Override
    public NDList dropout(NDArray input, float rate, boolean training) {
        return null;
    }

    @Override
    public NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta, int axis, float momentum, float eps, boolean training) {
        return null;
    }

    @Override
    public NDList gru(NDArray input, NDArray state, NDList params, boolean hasBiases, int numLayers, double dropRate, boolean training, boolean bidirectional, boolean batchFirst) {
        return null;
    }

    @Override
    public NDList lstm(NDArray input, NDList states, NDList params, boolean hasBiases, int numLayers, double dropRate, boolean training, boolean bidirectional, boolean batchFirst) {
        return null;
    }

    @Override
    public NDArray resize(int width, int height, int interpolation) {
        return null;
    }

    @Override
    public NDArray randomFlipLeftRight() {
        return null;
    }

    @Override
    public NDArray randomFlipTopBottom() {
        return null;
    }

    @Override
    public NDArray randomBrightness(float brightness) {
        return null;
    }

    @Override
    public NDArray randomHue(float hue) {
        return null;
    }

    @Override
    public NDArray randomColorJitter(float brightness, float contrast, float saturation, float hue) {
        return null;
    }

    @Override
    public NDArrayIndexer getIndexer() {
        return null;
    }

    @Override
    public NDArray where(NDArray condition, NDArray other) {
        return null;
    }

    @Override
    public NDArray stack(NDList arrays, int axis) {
        return null;
    }

    @Override
    public NDArray concat(NDList arrays, int axis) {
        return null;
    }

    @Override
    public NDList multiBoxTarget(NDList inputs, float iouThreshold, float ignoreLabel, float negativeMiningRatio, float negativeMiningThreshold, int minNegativeSamples) {
        return null;
    }

    @Override
    public NDList multiBoxPrior(List<Float> sizes, List<Float> ratios, List<Float> steps, List<Float> offsets, boolean clip) {
        return null;
    }

    @Override
    public NDList multiBoxDetection(NDList inputs, boolean clip, float threshold, int backgroundId, float nmsThreshold, boolean forceSuppress, int nmsTopK) {
        return null;
    }

    @Override
    public NDArray getArray() {
        return null;
    }
}
