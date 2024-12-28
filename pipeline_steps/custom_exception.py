class RetryablePipelineStepException(Exception):
    """
    Exception raised when a pipeline step fails and can be retried.
    """

    def __init__(self, message: str, retry_count: int = 3):
        self.message = message
        self.retry_count = retry_count
        super().__init__(message)
