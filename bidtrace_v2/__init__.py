# SPDX-License-Identifier: MPL-2.0
"""BidTrace original-stack document review application."""
from pkgutil import extend_path

# The installed core owns engine modules; the app supplies only its adapters.
# Launchers put the core first, and the shared namespace finds the app overlay.
__path__ = extend_path(__path__, __name__)
