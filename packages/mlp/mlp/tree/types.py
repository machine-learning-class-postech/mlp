type Tree[T] = T | dict[str, "Tree"] | list["Tree"]
