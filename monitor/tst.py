class Point3d():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        print_text = f'x:{self.x} y:{self.y} z:{self.z}'
        return print_text



a = Point3d(1,1,1)
print([a])