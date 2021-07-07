package org.apache.mxnet.integration.tests.jna;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.apache.mxnet.api.engine.Engine;
import org.apache.mxnet.api.util.PairList;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.jna.LibUtils;
import org.apache.mxnet.jna.MxnetLibrary;
import org.apache.mxnet.jna.NativeSizeByReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void loadLibraryFromCustomizePathTest() {
        MxnetLibrary lib = LibUtils.loadLibrary();
        IntBuffer version = IntBuffer.allocate(1);
        int ret = lib.MXGetVersion(version);
        assert ret == 0;
    }

    @Test
    public void loadNdArrayTest() {
        // load ndarray from file
        PairList<String, Pointer> pairList = JnaUtils.loadNdArrayFromFile("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params");

        Pointer handle = pairList.get(0).getValue();

        // save as byte array
        byte[] bytes = JnaUtils.saveNdArrayAsByteArray(handle);
        Pointer handle2 = JnaUtils.loadNdArrayFromByteArray(bytes, 0, bytes.length);
//        int ret = JnaUtils.loadNdArrayF2romByteArray(bytes, 0, 8);
//        assert p2.getIntArray(4, 3).equals(handle.getIntArray(4, 3));
        // save as byte buffer
        ByteBuffer byteBuffer = JnaUtils.saveNdArrayAsByteBuffer(handle);
        // save as PointerArray
//        PointerArray pa = JnaUtils.saveNdArrayAsPointArray(handle);

        // load ndarray from PointerArray
//        JnaUtils.loadNdArrayFromPointerArray(pa);
        // load ndarray from bytebuffer
        Pointer handle3 = JnaUtils.loadNdArrayFromByteBuffer(byteBuffer);

        logger.debug("pause");
    }

    @Test
    public void mxnetLibNdArrayLoadSaveFreeCallTest() {
        PairList<String, Pointer> pairList = JnaUtils.loadNdArrayFromFile("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params");
        PointerByReference outRef = new PointerByReference();
        NativeSizeByReference size = new NativeSizeByReference();
        JnaUtils.checkCall(JnaUtils.LIB.MXNDArraySaveRawBytes(pairList.get(0).getValue(), size, outRef));
        JnaUtils.checkCall(JnaUtils.LIB.MXNDArrayLoadFromRawBytes(outRef.getValue(), size.getValue(), outRef));
        JnaUtils.checkCall(JnaUtils.LIB.MXNDArrayFree(outRef.getValue()));
    }

    @Test
    public void loadNdArrayFromByteBufferTest() {

        try (
                FileInputStream is = new FileInputStream("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params");
                ByteArrayOutputStream baos = new ByteArrayOutputStream()
        ) {
//            byte[] bytes = new byte[102];
//            is.read(bytes, 0, 102);
            byte[] buffer = new byte[1024*4];
            int b = 0;
            while (-1 != (b = is.read(buffer))) {
                baos.write(buffer, 0, b);
            }
//            JnaUtils.loadNdArrayFromByteBuffer(baos.toByteArray(), 0, );
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
