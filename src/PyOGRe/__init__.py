import warnings

from PyOGRe.Calc import Calc, CovariantD, PartialD
from PyOGRe.Coordinates import new_coordinates
from PyOGRe.Documentation import __doc__, doc
from PyOGRe.Export import (export_all, import_all_from_file,
                           import_all_from_string, import_from_file,
                           import_from_string)
from PyOGRe.MathematicaParser.Interpreter import parse_str
from PyOGRe.Metric import new_metric
from PyOGRe.Options import (clear_instances, command_line_support,
                            delete_results, get_curve_parameter, get_instances,
                            get_options, jupyter_support, set_curve_parameter,
                            set_font_size, set_index_letters,
                            set_list_per_line, set_parallelization)
from PyOGRe.Tensor import new_tensor
from PyOGRe.Utils import (contract, elements, map_to_array, partial_contract,
                          str_symbols, zero_tensor)
from PyOGRe.Version import __version__, version

warnings.filterwarnings("ignore", category=DeprecationWarning)
