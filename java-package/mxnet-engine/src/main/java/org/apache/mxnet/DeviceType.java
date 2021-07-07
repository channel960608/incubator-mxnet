package org.apache.mxnet;


import org.apache.mxnet.api.Device;

/** {@code DeviceType} is a class used to map the Device name to their corresponding type number. */
public interface DeviceType {

    /**
     * Map device to its type number.
     *
     * @param device {@link Device} to map from
     * @return the number specified by engine
     */
    static int toDeviceType(Device device) {
        return 0;
    }

    /**
     * Map device to its type number.
     *
     * @param deviceType the number specified by engine
     * @return {@link Device} to map to
     */
    static String fromDeviceType(int deviceType) {
        return null;
    }
}
