import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from utils.provider_manager import ProviderManager

pm = ProviderManager()

response = pm.generate(
    "Reply with only OK"
)

print(response)