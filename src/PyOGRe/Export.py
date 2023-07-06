from typing import Any, Callable, Dict, List, Optional, Tuple

import sympy as sym
from IPython.display import Markdown, display

from PyOGRe.Christoffel import Christoffel
from PyOGRe.Coordinates import Coordinates
from PyOGRe.Geodesic import (GeodesicChristoffel, GeodesicLagrangian,
                             GeodesicTime)
from PyOGRe.Lagrangian import Lagrangian
from PyOGRe.Metric import Metric
from PyOGRe.OGReObject import OGReObject
from PyOGRe.Options import options
from PyOGRe.Tensor import Tensor

__doc__ = """
Export Module

PyOGRe is an Object-Oriented General Relativity Package for Python.
The full documentation is available at:
"""


def iterable_to_mathematica(
    iterable: Any
) -> str:
    """
    Convert an iterable to a string which can be evaluated in Mathematica.
    """

    if not isinstance(iterable, (tuple, list)):
        return str(sym.mathematica_code(iterable)) if not isinstance(iterable, str) else f'"{iterable}"'

    return "".join(
        [
            "{",
            ",".join(
                [
                    iterable_to_mathematica(i)
                    if not isinstance(i, str)
                    else f'"{i}"'
                    for i in iterable
                ]
            ),
            "}"
        ]
    )


def tensor_to_dict(
    tensor: OGReObject
) -> Dict[Any, Any]:
    """
    Creates a dictionary representing a tensor.
    """

    tensor_dict = {
        "Components": {
            (indices, coords._name): components
            for coords in tensor._components.keys()
            for indices, components in tensor._components[coords].items()
            if isinstance(coords, Coordinates)
        },
        "DefaultCoords": tensor._coordinates._name if isinstance(tensor, (Tensor, Metric)) else tensor._name,
        "DefaultIndices": tensor._indices,
        "Role": {
            "Tensor": "Tensor",
            "Metric": "Metric",
            "Coordinates": "Coordinates",
            "Lagrangian": "Lagrangian",
            "Christoffel": "Christoffel",
            "GeodesicTime": "GeodesicWithTimeParameter",
            "GeodesicChristoffel": "GeodesicFromChristoffel",
            "GeodesicLagrangian": "GeodesicFromLagrangian"
        }[tensor.__class__.__name__],
        "Symbol": tensor._symbol if tensor._symbol != "" else sym.Symbol("T")
    }

    if isinstance(tensor, (Tensor, Metric)):
        tensor_dict["Metric"] = tensor._metric._name

    if isinstance(tensor, Coordinates):
        if tensor._jacobians != {}:
            tensor_dict["Jacobians"] = {
                coords._name: {
                    "Jacobian": tensor._jacobians[coords],
                    "InverseJacobian": tensor._inverse_jacobians[coords],
                    "ChristoffelJacobian": tensor._christoffel_jacobians[coords]
                }
                for coords in tensor._jacobians.keys()
            }

        tensor_dict["CoordTransformations"] = {
            coords._name: {
                equation.lhs: equation.rhs
                for equation in tensor._transformations[coords]
                if equation is not None
            }
            for coords in tensor._transformations.keys()
        }

    return tensor_dict


def release_hold(
    string: str
) -> str:
    """
    Release hold of a string.
    """

    return string + r"\\ReleaseHold"


def tensor_dict_to_association(
    d: Dict[Any, Any],
    brackets: bool = True,
    sub_brackets: bool = True
) -> str:
    """
    Convert a tensor dictionary to an association for Mathematica.
    """

    return "".join(
        [
            "<|" if brackets else "{",
            ",".join(
                [
                    "{}->{}".format(
                        iterable_to_mathematica(k),
                        f'"{v}"'
                        if isinstance(v, str)
                        else release_hold(
                            sym.mathematica_code(
                                [v[()]] if isinstance(v, sym.Array) and v.rank() == 0 else v
                            )
                        )
                    )
                    if not isinstance(v, dict)
                    else "{}->{}".format(
                        iterable_to_mathematica(k),
                        tensor_dict_to_association(v, brackets=sub_brackets)
                        if k != "CoordTransformations"
                        else tensor_dict_to_association(v, sub_brackets=False)
                    )
                    for k, v in d.items()
                ]
            ),
            "|>" if brackets else "}"
        ]
    )


def string_to_tensor(
    string: str
) -> List[Tuple[Callable[[Any], Any], Dict[str, Any]]]:
    """
    Creates a Tensor from a string.
    """
    if not isinstance(string, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'string', got '{type(string)}'."
        )

    from PyOGRe.MathematicaParser.Interpreter import parse_str
    tensor_data = parse_str(string)

    tensor_list = []
    for tensor_name, tensor in tensor_data.items():
        if tensor_name == "Options":
            print(tensor)
            continue

        role = tensor["Role"]

        PyOGRe_class: Callable[[Any], Any] = {
            'Coordinates': Coordinates,
            'Metric': Metric,
            'Tensor': Tensor,
            'Calculated': Tensor,
            'Lagrangian': Lagrangian,
            'Christoffel': Christoffel,
            'GeodesicFromChristoffel': GeodesicChristoffel,
            'GeodesicFromLagrangian': GeodesicLagrangian,
            'GeodesicWithTimeParameter': GeodesicTime,
            'NormSquared': Tensor,
            'Riemann': Tensor,
            'Ricci Tensor': Tensor,
            'Ricci Scalar': Tensor,
            'Einstein': Tensor,
        }[role]

        coordinates = tensor['DefaultCoords']

        indices = tensor['DefaultIndices']

        configurations, representations = tensor["Components"]
        components = None
        for config, representation in zip(configurations, representations):
            if config[0] == indices and config[1] == coordinates:
                components = sym.Array(representation) if len(representation) > 1 else sym.Array(representation[0])
        if components is None:
            raise Exception("Could not valid find components.")

        kwargs: Dict[str, Any] = {
            "name": tensor_name,
            "indices": tuple(indices),
            "components": components,
        }
        if 'Symbol' in tensor:
            kwargs["symbol"] = tensor['Symbol']

        if PyOGRe_class is not Coordinates:  # type: ignore[comparison-overlap]
            kwargs["coords"] = coordinates

        if PyOGRe_class not in [Coordinates, Metric]:
            exists = False
            for instance in OGReObject._instances:
                if instance._name == tensor['Metric']:
                    exists = True
                    kwargs["metric"] = instance
            if not exists:
                kwargs["metric"] = tensor['Metric']

        if "CoordTransformations" in tensor:
            coord_transformations = {}
            for coords, rules in tensor["CoordTransformations"].items():
                exists = False
                for instance in OGReObject._instances:
                    if instance._name == coords:
                        exists = True
                        coord_transformations[instance] = rules
                if not exists:
                    coord_transformations[coords] = rules
            if coord_transformations:
                kwargs["transformations"] = coord_transformations

        tensor_list.append((PyOGRe_class, kwargs))

    return tensor_list


def export_to_string(
    tensor: OGReObject
) -> str:
    """
    Exports a tensor to a string.
    """
    from PyOGRe.Version import __version__
    tensor_dict = tensor_to_dict(tensor)
    tensor_dict["OGReVersion"] = f"PyOGRe v{__version__}"

    return "".join(
        [
            "<|",
            f'"{tensor._name}"',
            "->",
            tensor_dict_to_association(tensor_dict),
            "|>"
        ]
    )


def export_to_file(
    tensor: OGReObject,
    filename: str
) -> str:
    """
    Exports a tensor to a file.
    """

    tensor_string = export_to_string(tensor)
    with open(filename, "w") as f:
        f.write(tensor_string)
    return f"Exported {tensor._name} to {filename}"


def export_all_to_string(
) -> str:
    """
    Exports all tensors to a string.
    """
    from PyOGRe.Version import __version__
    return "".join(
        [
            "<|",
            f'"Options"-><|"OGReVersion"->"PyOGRe v{__version__}"|>,',
            ",".join(
                [
                    f'"{tensor._name}"->{tensor_dict_to_association(tensor_to_dict(tensor))}'
                    for tensor in OGReObject._instances
                ]
            ),
            "|>"
        ]
    )


def export_all_to_file(
    filename: str
) -> str:
    """
    Exports all tensors to a file.
    """

    all_tensor_strings = export_all_to_string()
    with open(filename, "w") as f:
        f.write(all_tensor_strings)
    return f"Exported all tensors to {filename}"


def export_all(
    filename: Optional[str] = None
) -> Optional[str]:
    """
    Exports all tensors to a string.

    Supplying a filepath as a string will write the tensor data to a file.
    """

    if filename is not None and not isinstance(filename, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'filename', got '{type(filename)}'."
        )

    if filename is None:
        return export_all_to_string()

    if filename is not None:
        status = export_all_to_file(filename)
        if options.LATEX:
            font_size = options.FONT_SIZE
            display(Markdown(
                f"<div align=left style='font-size:{font_size}pt'> \n\n" +
                status +
                "\n\n </div>"
            ))
        if not options.LATEX:
            print(status)

    return None


def import_from_string(
    string: str
) -> OGReObject:
    """
    Imports a tensor from a string.
    """

    tensor_class, kwargs = string_to_tensor(string)[0]

    if "coords" in kwargs:
        for instance in OGReObject._instances:
            if instance._name == kwargs["coords"]:
                kwargs["coords"] = instance
                break
        else:
            raise Exception(f"Could not find coordinate system {kwargs['coords']}.")

    return tensor_class(**kwargs)  # type: ignore[call-arg, no-any-return]


def import_from_file(
    filename: str
) -> OGReObject:
    """
    Imports a tensor from a file.
    """

    if not isinstance(filename, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'filename', got '{type(filename)}'."
        )

    with open(filename, "r") as f:
        string = f.read()

    return import_from_string(string)


def import_all_from_string(
    string: str
) -> List[OGReObject]:
    """
    Imports all tensors from a string.
    """

    if not isinstance(string, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'string', got '{type(string)}'."
        )

    from PyOGRe.Options import clear_instances
    clear_instances()

    tensor_list = string_to_tensor(string)

    for tensor_class, kwargs in tensor_list:
        if tensor_class == Coordinates:  # type: ignore[comparison-overlap]
            new_kwargs = kwargs.copy()
            if "transformations" in kwargs:
                del new_kwargs["transformations"]
            tensor_class(**new_kwargs)  # type: ignore[call-arg]

    for tensor_class, kwargs in tensor_list:
        if "coords" in kwargs and isinstance(kwargs["coords"], str):
            for instance in OGReObject._instances:
                if instance._name == kwargs["coords"]:
                    kwargs["coords"] = instance

    for tensor_class, kwargs in tensor_list:
        if tensor_class == Coordinates and "transformations" in kwargs:  # type: ignore[comparison-overlap]
            for instance in OGReObject._instances:
                if instance._name == kwargs["name"] and isinstance(instance, Coordinates):
                    for key, value in kwargs["transformations"].items():
                        for other_instance in OGReObject._instances:
                            if other_instance._name == key:
                                instance.add_coord_transformation(coords=other_instance, rules=value)  # type: ignore[arg-type]
                                break
                        else:
                            raise Exception(f"Could not find coordinate system {kwargs['coords']}.")

    for tensor_class, kwargs in tensor_list:
        if "coords" in kwargs and isinstance(kwargs["coords"], str):
            for instance in OGReObject._instances:
                if instance._name == kwargs["coords"]:
                    kwargs["coords"] = instance
                    break
            else:
                raise Exception(f"Could not find coordinate system {kwargs['coords']}.")

    for tensor_class, kwargs in tensor_list:
        if tensor_class is Metric:  # type: ignore[comparison-overlap]
            tensor_class(**kwargs)  # type: ignore[call-arg]

    for tensor_class, kwargs in tensor_list:
        if "metric" in kwargs and isinstance(kwargs["metric"], str):
            for instance in OGReObject._instances:
                if instance._name == kwargs["metric"]:
                    kwargs["metric"] = instance
                    break
            else:
                raise Exception(f"Could not find metric {kwargs['metric']}.")

    for tensor_class, kwargs in tensor_list:
        if tensor_class not in [Coordinates, Metric]:
            tensor_class(**kwargs)  # type: ignore[call-arg]

    return OGReObject._instances


def import_all_from_file(
    filename: str
) -> List[OGReObject]:
    """
    Imports all tensors from a file.
    """

    from PyOGRe.Options import clear_instances
    clear_instances()

    if not isinstance(filename, str):
        raise TypeError(
            f"Expected type '{str}' for argument 'filename', got '{type(filename)}'."
        )

    with open(filename, "r") as f:
        string = f.read()

    return import_all_from_string(string)
