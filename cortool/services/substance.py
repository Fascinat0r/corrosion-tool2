from cortool.models.substance import Substance, substance_dict


def create_substance(name: str) -> Substance:
    """ Функция создания объекта Substance на основе строки. """
    if name in substance_dict:
        return substance_dict[name]()
    else:
        raise ValueError(f"No substance class defined for {name}")
