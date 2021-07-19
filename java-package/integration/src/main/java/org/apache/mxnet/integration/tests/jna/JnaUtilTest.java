package org.apache.mxnet.integration.tests.jna;

import org.apache.mxnet.engine.BaseMxResource;
import org.apache.mxnet.engine.Device;
import org.apache.mxnet.engine.MxResource;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.apache.mxnet.ndarray.MxNDArray;
import org.apache.mxnet.ndarray.MxNDList;
import org.apache.mxnet.ndarray.types.Shape;
import org.apache.mxnet.nn.MxSymbolBlock;
import org.apache.mxnet.nn.Parameter;
import org.apache.mxnet.training.ParameterStore;
import org.apache.mxnet.util.PairList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JnaUtilTest {

    private static final Logger logger = LoggerFactory.getLogger(JnaUtilTest.class);

    @Test
    public void doForwardTest() {
        // TODO: replace the Path of model with soft decoding
        try (
                MxResource base = BaseMxResource.getSystemMxResource()
                ) {
            Symbol symbol  = Symbol.loadFromFile(base,
                    "/Users/cspchen/.djl.ai/cache/repo/model/cv/image_classification/ai/djl/mxnet/mlp/mnist/0.0.1/mlp-symbol.json");
            MxSymbolBlock block = new MxSymbolBlock(base, symbol);
            Device device = Device.defaultIfNull();
            MxNDList mxNDArray = JnaUtils.loadNdArray(
                    base,
                    Paths.get("/Users/cspchen/.djl.ai/cache/repo/model/cv/image_classification/ai/djl/mxnet/mlp/mnist/0.0.1/mlp-0000.params"),
                    Device.defaultIfNull(null));

            // load parameters
            List<Parameter> parameters = block.getAllParameters();
            Map<String, Parameter> map = new LinkedHashMap<>();
            parameters.forEach(p -> map.put(p.getName(), p));

            for (MxNDArray nd : mxNDArray) {
                String key = nd.getName();
                if (key == null) {
                    throw new IllegalArgumentException("Array names must be present in parameter file");
                }

                String paramName = key.split(":", 2)[1];
                Parameter parameter = map.remove(paramName);
                parameter.setArray(nd);
            }
            block.setInputNames(new ArrayList<>(map.keySet()));

            MxNDArray arr = MxNDArray.create(base, new Shape(1, 28, 28), device).ones();
            block.forward(new ParameterStore(base, false, device), new MxNDList(arr), false, new PairList<>(), device);
            System.out.println(base.getSubResource().size());
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
            throw e;
        }
        System.out.println("Ok");
        System.out.println(BaseMxResource.getSystemMxResource().getSubResource().size());

    }

    @Test
    public void createNdArray() {
        try {
            try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                int[] originIntegerArray = new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                float[] originFlaotArray = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                double[] originDoubleArray = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                long[] originLongArray = new long[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                boolean[] originBooleanArray = new boolean[]{true, false, false, true, true, true, true, false, false, true, true, true};
                byte[] originByteArray = new byte[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
                MxNDArray intArray = MxNDArray.create(base, originIntegerArray, new Shape(3, 4));
                MxNDArray floatArray = MxNDArray.create(base, originFlaotArray, new Shape(3, 4));
                MxNDArray doubleArray = MxNDArray.create(base, originDoubleArray, new Shape(3, 4));
                MxNDArray longArray = MxNDArray.create(base, originLongArray, new Shape(3, 4));
                MxNDArray booleanArray = MxNDArray.create(base, originBooleanArray, new Shape(3, 4));
                MxNDArray byteArray = MxNDArray.create(base, originByteArray, new Shape(3, 4));
                MxNDArray intArray2 = MxNDArray.create(base, originIntegerArray);
                MxNDArray floatArray2 = MxNDArray.create(base, originFlaotArray);
                MxNDArray doubleArray2 = MxNDArray.create(base, originDoubleArray);
                MxNDArray longArray2 = MxNDArray.create(base, originLongArray);
                MxNDArray booleanArray2 = MxNDArray.create(base, originBooleanArray);
                MxNDArray byteArray2 = MxNDArray.create(base, originByteArray);

                Integer[] ndArrayInt = (Integer[]) intArray.toArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt);
                // Float -> Double
                double[] floats = Arrays.stream(floatArray.toArray()).mapToDouble(Number::floatValue).toArray();
                Assert.assertEquals(originDoubleArray, floats);
                Double[] ndArrayDouble = (Double[]) doubleArray.toArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble);
                Long[] ndArrayLong = (Long[]) longArray.toArray();
                Assert.assertEquals(originLongArray, ndArrayLong);
                boolean[] ndArrayBoolean = booleanArray.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean);
                byte[] ndArrayByte = byteArray.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte);


                Integer[] ndArrayInt2 = (Integer[]) intArray2.toArray();
                Assert.assertEquals(originIntegerArray, ndArrayInt2);

                // Float -> Double
                double[] floats2 = Arrays.stream(floatArray2.toArray()).mapToDouble(Number::floatValue).toArray();
                Assert.assertEquals(originDoubleArray, floats2);
                Double[] ndArrayDouble2 = (Double[]) doubleArray2.toArray();
                Assert.assertEquals(originDoubleArray, ndArrayDouble2);
                Long[] ndArrayLong2 = (Long[]) longArray2.toArray();
                Assert.assertEquals(originLongArray, ndArrayLong2);
                boolean[] ndArrayBoolean2 = booleanArray2.toBooleanArray();
                Assert.assertEquals(originBooleanArray, ndArrayBoolean2);
                byte[] ndArrayByte2 = byteArray2.toByteArray();
                Assert.assertEquals(originByteArray, ndArrayByte2);
            } catch (Exception e) {
                logger.error(e.getMessage());
                e.printStackTrace();
                throw e;
            }
            BaseMxResource base = BaseMxResource.getSystemMxResource();
//            assert base.getSubResource().size() == 0;
        } catch (Exception e) {
            logger.error(e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }

    @Test
    public void loadNdArray() {

        try (BaseMxResource base = BaseMxResource.getSystemMxResource()) {
                MxNDList mxNDArray = JnaUtils.loadNdArray(
                        base,
                        Paths.get("/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-0000.params"),
                        Device.defaultIfNull(null));

            System.out.println(base.getSubResource().size());
        }
        System.out.println(BaseMxResource.getSystemMxResource().getSubResource().size());

    }
}
