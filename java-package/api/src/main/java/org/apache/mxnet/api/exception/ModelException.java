package org.apache.mxnet.api.exception;

public class ModelException extends BaseException{

//    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new exception with the specified detail message. The cause is not initialized,
     * and may subsequently be initialized by a call to {@link #initCause}.
     *
     * @param message the detail message that is saved for later retrieval by the {@link
     *     #getMessage()} method
     */
    public ModelException(String message) {
        super(message);
    }

    /**
     * Constructs a new exception with the specified detail message and cause.
     *
     * <p>Note that the detail message associated with {@code cause} is <i>not</i> automatically
     * incorporated in this exception's detail message.
     *
     * @param message the detail message that is saved for later retrieval by the {@link
     *     #getMessage()} method
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public ModelException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new exception with the specified cause and a detail message of {@code
     * (cause==null ? null : cause.toString())} which typically contains the class and detail
     * message of {@code cause}. This constructor is useful for exceptions that are little more than
     * wrappers for other throwables. For example, {@link java.security.PrivilegedActionException}.
     *
     * @param cause the cause that is saved for later retrieval by the {@link #getCause()} method. A
     *     {@code null} value is permitted, and indicates that the cause is nonexistent or unknown
     */
    public ModelException(Throwable cause) {
        super(cause);
    }

}