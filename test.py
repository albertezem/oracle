import warnings
from statsbombpy.api_client import NoAuthWarning

warnings.simplefilter("ignore", NoAuthWarning)

from socceraction.data.statsbomb import StatsBombLoader

api = StatsBombLoader(getter="remote", creds=None)
