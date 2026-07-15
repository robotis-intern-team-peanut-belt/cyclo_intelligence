# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset preparation: download, convert v2.1 -> v3.0, trim mobile dims.

Submodules that need numpy/pandas import them inside functions, so that
`import cyclo_train` stays pyyaml-only and keeps working in every image.
"""

from .errors import DataPrepError

__all__ = ["DataPrepError"]
