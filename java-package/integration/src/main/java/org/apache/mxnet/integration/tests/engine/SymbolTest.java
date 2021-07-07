package org.apache.mxnet.integration.tests.engine;

import org.apache.mxnet.api.exception.EngineException;
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
        try (Symbol symbol =
                     Symbol.loadFromFile(
                             "/Users/cspchen/Downloads/mxnet_resnet18/resnet18_v1-symbol.json")) {
            String strSymbol = JnaUtils.printSymbol(symbol.getHandle());
            String[] strs = JnaUtils.listSymbolOutputs(symbol.getHandle());
            assert true;
        } catch (EngineException e) {
            e.printStackTrace();
            throw e;
        }
    }
}
