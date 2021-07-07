package org.apache.mxnet.api.util;


/**
 * Static convenience methods that help a method or constructor check whether it was invoked
 * correctly.
 */
public final class Preconditions {

    private Preconditions() {}

    /**
     * Ensures the truth of an expression involving one or more parameters to the calling method.
     *
     * @param expression a boolean expression
     * @param errorMessage the exception message to use if the check fails
     */
    public static void checkArgument(boolean expression, String errorMessage) {
        if (!expression) {
            throw new IllegalArgumentException(errorMessage);
        }
    }
}
