# raytracer/canvas/canvas.py

from raytracer.colors import Color


class Canvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pixels = [[Color(0, 0, 0) for _ in range(width)] for _ in range(height)]

    def write_pixel(self, x: int, y: int, color: Color) -> None:
        self.pixels[y][x] = color

    def pixel_at(self, x: int, y: int) -> Color:
        return self.pixels[y][x]

    def to_ppm(self) -> str:
        header = f"P3\n{self.width} {self.height}\n255\n"
        body = ""
        for row in self.pixels:
            for pixel in row:
                body += f"{self._to_ppm_color(pixel)}"
            body = body[0:-1] + "\n"
        return header + body

    def _to_ppm_color(self, color: Color) -> str:
        def clamp(x: float) -> int:
            if x > 1:
                return 255
            if x < 0:
                return 0
            return round(x * 255)

        return f"{clamp(color.x)} {clamp(color.y)} {clamp(color.z)} "
