# Hugsim Client

## Quick Start

```python
from hugsim_client import HugsimClient

client = HugsimClient()

# Get the current state of the environment
client.get_current_state()

# Reset the environment
client.reset_env()

# Execute an action in the environment, it return current state.
client.execute_action(xxx)
```
