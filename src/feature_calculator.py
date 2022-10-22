# Mock code
# Feature calculation is not necessary
import logging

logger = logging.getLogger(__name__)


class FeatureCalculatorMock:
    def __init__(self, *args, **kwargs):
        """Mock class -- please ignore."""
        logger.warning("Feature calculation is disabled in this source code.")
        pass

    def skip(self):
        feature = {
            "none": {"per_event": True, "value": None},
        }
        return feature

    def calculate_feature(self, *args, skip: bool = False, **kwargs) -> dict:
        """Mock function."""
        return self.skip()
