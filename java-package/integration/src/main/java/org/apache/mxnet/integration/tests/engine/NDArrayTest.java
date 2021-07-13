package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.ndarray.types.Shape;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxNDArray;
import org.apache.mxnet.engine.MxNDManager;
import org.apache.mxnet.engine.MxSymbolBlock;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.jna.JnaUtils;
import org.testng.annotations.Test;

public class NDArrayTest {

    @Test
    public void ndarrayTest() {
        try (MxNDManager manager = MxNDManager.getSystemManager();
             Symbol symbol =
                     Symbol.loadFromFile(manager, "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json")) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());
            MxNDArray mxNDArray = (MxNDArray) manager.zeros(new Shape(new long[]{5, 1}));
            MxSymbolBlock mxSymbolBlock = new MxSymbolBlock(manager, symbol);
            CachedOp cachedOp = JnaUtils.createCachedOp(mxSymbolBlock, manager, false);
            System.out.println(1);
        } catch (EngineException e) {
            e.printStackTrace();
            throw e;
        }
    }


}
