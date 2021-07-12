package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.engine.CachedOp;
import org.apache.mxnet.engine.MxNDManager;
import org.apache.mxnet.engine.MxSymbolBlock;
import org.apache.mxnet.engine.Symbol;
import org.apache.mxnet.integration.tests.jna.JnaUtilTest;
import org.apache.mxnet.jna.JnaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

public class SymbolTest {

    private static final Logger logger = LoggerFactory.getLogger(SymbolTest.class);

    @Test
    public void loadAndCloseTest() {
        try (NDManager manager = NDManager.newBaseManager();
             Symbol symbol =
                     Symbol.loadFromFile((MxNDManager) manager, "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json")) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());
            assert true;
        } catch (EngineException e) {
            e.printStackTrace();
            throw e;
        }
    }


    @Test
    public void loadCachedOp() {
        try (NDManager manager = NDManager.newBaseManager();
             Symbol symbol =
                     Symbol.loadFromFile((MxNDManager) manager,
                             "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json")
             ) {
            MxSymbolBlock mxSymbolBlock = new MxSymbolBlock(manager, symbol);

            CachedOp cachedOp = JnaUtils.createCachedOp(mxSymbolBlock, (MxNDManager) manager, false);
            cachedOp.close();
        } catch (EngineException e) {
            e.printStackTrace();
            throw e;
        } finally {

        }
    }


}
