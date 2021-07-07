package org.apache.mxnet.api.ndarray;

public interface LazyNDArray extends NDArray {

    /** Runs the current NDArray and sleeps until the value is ready to read. */
    void waitToRead();

    /** Runs the current NDArray and sleeps until the value is ready to write. */
    void waitToWrite();

    /** Runs all NDArrays and sleeps until their values are fully computed. */
    void waitAll();

}
