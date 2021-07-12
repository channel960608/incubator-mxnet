package org.apache.mxnet.api.engine;

import org.apache.mxnet.api.Device;
import org.apache.mxnet.api.exception.EngineException;
import org.apache.mxnet.api.ndarray.NDManager;
import org.apache.mxnet.api.nn.SymbolBlock;
import org.apache.mxnet.api.util.Utils;
import org.apache.mxnet.api.util.cuda.CudaUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

public abstract class Engine {

    private static final Logger logger = LoggerFactory.getLogger(Engine.class);

    private static final Map<String, EngineProvider> ALL_ENGINES = new ConcurrentHashMap<>();

    private static final String DEFAULT_ENGINE = initEngine();

    private Device defaultDevice;

    // use object to check if it's set
    private Integer seed;

    /**
     * Returns the name of the Engine.
     *
     * @return the name of the engine
     */
    public abstract String getEngineName();

    public static Engine getInstance() {
        if (DEFAULT_ENGINE == null) {
            throw new EngineException(
                    "No deep learning engine found."
                            + System.lineSeparator()
                            + "Please refer to https://github.com/deepjavalibrary/djl/blob/master/docs/development/troubleshooting.md for more details.");
        }
        return getEngine(System.getProperty("ai.djl.default_engine", DEFAULT_ENGINE));
    }

    /**
     * Returns the {@code Engine} with the given name.
     *
     * @param engineName the name of Engine to retrieve
     * @return the instance of {@code Engine}
     * @see EngineProvider
     */
    public static Engine getEngine(String engineName) {
        EngineProvider provider = ALL_ENGINES.get(engineName);
        if (provider == null) {
            throw new IllegalArgumentException("Deep learning engine not found: " + engineName);
        }
        return provider.getEngine();
    }

    private static synchronized String initEngine() {
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        for (EngineProvider provider : loaders) {
            logger.debug("Found EngineProvider: {}", provider.getEngineName());
            ALL_ENGINES.put(provider.getEngineName(), provider);
        }

        if (ALL_ENGINES.isEmpty()) {
            logger.debug("No engine found from EngineProvider");
            return null;
        }

        String defaultEngine = System.getenv("DJL_DEFAULT_ENGINE");
        defaultEngine = System.getProperty("ai.djl.default_engine", defaultEngine);
        if (defaultEngine == null || defaultEngine.isEmpty()) {
            int rank = Integer.MAX_VALUE;
            for (EngineProvider provider : ALL_ENGINES.values()) {
                if (provider.getEngineRank() < rank) {
                    defaultEngine = provider.getEngineName();
                    rank = provider.getEngineRank();
                }
            }
        } else if (!ALL_ENGINES.containsKey(defaultEngine)) {
            throw new EngineException("Unknown default engine: " + defaultEngine);
        }
        logger.debug("Found default engine: {}", defaultEngine);
        return defaultEngine;
    }

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return {@code true} if the engine has the specified capability
     */
    public abstract boolean hasCapability(String capability);

    /**
     * Returns the engine's default {@link Device}.
     *
     * @return the engine's default {@link Device}
     */
    public Device defaultDevice() {
        if (defaultDevice == null) {
            if (hasCapability(StandardCapabilities.CUDA) && CudaUtils.getGpuCount() > 0) {
                defaultDevice = Device.gpu();
            } else {
                defaultDevice = Device.cpu();
            }
        }
        return defaultDevice;
    }

    /**
     * Creates a new top-level {@link NDManager}.
     *
     * <p>{@code NDManager} will inherit default {@link Device}.
     *
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager();

    /**
     * Creates a new top-level {@link NDManager} with specified {@link Device}.
     *
     * @param device the default {@link Device}
     * @return a new top-level {@code NDManager}
     */
    public abstract NDManager newBaseManager(Device device);

    /**
     * Construct an empty SymbolBlock for loading.
     *
     * @param manager the manager to manage parameters
     * @return Empty {@link SymbolBlock} for static graph
     */
    public abstract SymbolBlock newSymbolBlock(NDManager manager);

    /** Prints debug information about the environment for debugging environment issues. */
    @SuppressWarnings("PMD.SystemPrintln")
    public static void debugEnvironment() {
        System.out.println("----------- System Properties -----------");
        System.getProperties().forEach((k, v) -> System.out.println(k + ": " + v));

        System.out.println();
        System.out.println("--------- Environment Variables ---------");
        System.getenv().forEach((k, v) -> System.out.println(k + ": " + v));

        System.out.println();
        System.out.println("-------------- Directories --------------");
        try {
            Path temp = Paths.get(System.getProperty("java.io.tmpdir"));
            System.out.println("temp directory: " + temp);
            Path tmpFile = Files.createTempFile("test", ".tmp");
            Files.delete(tmpFile);

            Path cacheDir = Utils.getCacheDir();
            System.out.println("DJL cache directory: " + cacheDir.toAbsolutePath());

            Path path = Utils.getEngineCacheDir();
            System.out.println("Engine cache directory: " + path.toAbsolutePath());
            Files.createDirectories(path);
            if (!Files.isWritable(path)) {
                System.out.println("Engine cache directory is not writable!!!");
            }
        } catch (Throwable e) {
            e.printStackTrace(System.out);
        }

        System.out.println();
        System.out.println("------------------ CUDA -----------------");
        int gpuCount = Device.getGpuCount();
        System.out.println("GPU Count: " + gpuCount);
        System.out.println("Default Device: " + Device.defaultDevice());
        if (gpuCount > 0) {
            System.out.println("CUDA: " + CudaUtils.getCudaVersionString());
            System.out.println("ARCH: " + CudaUtils.getComputeCapability(0));
        }
        for (int i = 0; i < gpuCount; ++i) {
            Device device = Device.gpu(i);
            MemoryUsage mem = CudaUtils.getGpuMemory(device);
            System.out.println("GPU(" + i + ") memory used: " + mem.getCommitted() + " bytes");
        }

        System.out.println();
        System.out.println("----------------- Engines ---------------");
        System.out.println("Default Engine: " + DEFAULT_ENGINE);
        for (EngineProvider provider : ALL_ENGINES.values()) {
            System.out.println(provider.getEngineName() + ": " + provider.getEngineRank());
            try {
                provider.getEngine();
            } catch (EngineException e) {
                e.printStackTrace(System.out);
            }
        }
    }
}
