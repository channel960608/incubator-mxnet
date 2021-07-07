package org.apache.mxnet.integration.tests.api;

import com.sun.jna.Pointer;
import org.apache.mxnet.api.ndarray.NDArray;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.integration.tests.jna.JnaUtilTest;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

public class NDArrayTest {

//    private static final Logger logger = LoggerFactory.getLogger(NDArrayTest.class);
    @Test
    public void createNdArrayTest() {

        PairList<String, Pointer> pairList = JnaUtils.loadNdArrayFromFile("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params");

        Pointer handle = pairList.get(0).getValue();

        MxNDArray nDArray = MxNDArray.create(handle);

        nDArray.toString();
        System.out.println("ok");
    }


}
