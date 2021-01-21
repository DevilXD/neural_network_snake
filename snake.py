from __future__ import annotations

import random
from copy import copy
from time import sleep
from typing import Optional, Union, Tuple, NamedTuple, Callable

import pyglet
from pyglet.text import Label
from pyglet.window import key
from pyglet.graphics import Batch
from pyglet.image import SolidColorImagePattern
from pyglet.window import Window as OriginalWindow
from pyglet.sprite import Sprite as OriginalSprite

Index = Union[int, slice]
Color = Tuple[int, int, int, int]


def slice_to_range(s, l):
    return range(*s.indices(l))


def center(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


class PixelArray:
    def __init__(self, image: pyglet.image.ImageData):
        if image.format != "RGBA":
            raise TypeError("Only RGBA images allowed!")
        self._image = image
        self._image_data = bytearray()

    def __enter__(self):
        self._image_data = bytearray(self._image.get_data())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._image.set_data(self._image.format, self._image.pitch, bytes(self._image_data))

    def __getitem__(self, index: Union[int, slice]):
        # TODO: Implement this
        return NotImplemented

    def __setitem__(self, index: Union[Index, Tuple[Index, Index]], value: Color):
        if isinstance(index, int):
            for i, v in enumerate(value):
                self._image_data[index + i] = v
        if isinstance(index, slice):
            for p in slice_to_range(index, self._image.width * self._image.height):
                for i, v in enumerate(value):
                    self._image_data[p + i] = v
        elif isinstance(index, tuple):
            rxi, ryi = index
            if isinstance(rxi, int):
                rx = (rxi % self._image.width,)
            elif isinstance(rxi, slice):
                rx = slice_to_range(rxi, self._image.width)
            else:
                raise KeyError
            if isinstance(ryi, int):
                ry = (ryi % self._image.height,)
            elif isinstance(ryi, slice):
                ry = slice_to_range(ryi, self._image.height)
            else:
                raise KeyError
            for x in rx:
                for y in ry:
                    idx = (x + y * self._image.width) * 4
                    for i, v in enumerate(value):
                        self._image_data[idx + i] = v
        else:
            raise KeyError


class Position(NamedTuple):
    x: int
    y: int


class Window(OriginalWindow):
    def get_size(self):
        return Position(*super().get_size())


class Sprite(OriginalSprite):
    def __init__(self, image, *args, rotation=0, **kwargs):
        super().__init__(image.get_texture().get_transform(rotate=rotation), *args, **kwargs)

    def move_and_turn(self, dx: int = 0, dy: int = 0, rotation: Optional[int] = None):
        if rotation is not None:
            self.update(x=self.x + dx, y=self.y + dy, rotation=rotation)
        else:
            self.update(x=self.x + dx, y=self.y + dy)

    def bounding_box(self) -> Tuple[Position, Position]:
        return (
            Position(
                self.x - self.image.anchor_x,
                self.y - self.image.anchor_y,
            ),
            Position(
                self.x - self.image.anchor_x + self.width,
                self.y - self.image.anchor_y + self.height,
            ),
        )

    def overlaps(self, other: Union[Sprite, Window], *, fully: bool = False):
        p1, p2 = self.bounding_box()
        if isinstance(other, Sprite):
            p3, p4 = other.bounding_box()
        else:
            p3 = Position(0, 0)
            p4 = other.get_size()
        if p2.x > p3.x and p4.x > p1.x and p2.y > p3.y and p4.y > p1.y:
            if fully:
                if (
                    p2.x >= p4.x and p2.y >= p4.y and p3.x >= p1.x and p3.y >= p1.y
                    or p4.x >= p2.x and p4.y >= p2.y and p1.x >= p3.x and p1.y >= p3.y
                ):
                    return True
                return False
            return True
        return False


# make the apple
apple_img = SolidColorImagePattern((0, 255, 0, 255)).create_image(10, 10)
# round the apple
with PixelArray(apple_img) as apple_array:
    apple_array[0:3, 0]  = (0, 0, 0, 0)
    apple_array[0:2, 1]  = (0, 0, 0, 0)
    apple_array[0, 2]    = (0, 0, 0, 0)
    apple_array[7:10, 0] = (0, 0, 0, 0)
    apple_array[8:10, 1] = (0, 0, 0, 0)
    apple_array[9:10, 2] = (0, 0, 0, 0)
    apple_array[0, 7]    = (0, 0, 0, 0)
    apple_array[0:2, 8]  = (0, 0, 0, 0)
    apple_array[0:3, 9]  = (0, 0, 0, 0)
    apple_array[9, 7]    = (0, 0, 0, 0)
    apple_array[8:10, 8] = (0, 0, 0, 0)
    apple_array[7:10, 9] = (0, 0, 0, 0)
# center the apple
center(apple_img)

# make a snake segment
snake_seg  = SolidColorImagePattern((255, 0, 0, 255)).create_image(20, 20)
# create 5 copies
snake_head = copy(snake_seg)
snake_body = copy(snake_seg)
snake_cor1 = copy(snake_seg)
snake_cor2 = copy(snake_seg)
snake_tail = copy(snake_seg)

# style the head
with PixelArray(snake_head) as seg:
    # thin and barb sides
    seg[0, :]    = (0, 0, 0, 0)
    seg[1, ::2]  = (0, 0, 0, 0)
    seg[-2, ::2] = (0, 0, 0, 0)
    seg[-1:, :]  = (0, 0, 0, 0)
    # round the front
    seg[1:3, -3:]   = (0, 0, 0, 0)
    seg[3, -2:]     = (0, 0, 0, 0)
    seg[4:6, -1]    = (0, 0, 0, 0)
    seg[-3:-1, -3:] = (0, 0, 0, 0)
    seg[-4, -1:]    = (0, 0, 0, 0)
    seg[-5:-3, -1]  = (0, 0, 0, 0)
    # left eye
    seg[4, 4:6]   = (0, 0, 0, 0)
    seg[5:7, 3:7] = (0, 0, 0, 0)
    seg[7, 4:6]   = (0, 0, 0, 0)
    # right eye
    seg[13, 4:6]    = (0, 0, 0, 0)
    seg[14:16, 3:7] = (0, 0, 0, 0)
    seg[16, 4:6]    = (0, 0, 0, 0)
    # mouth
    seg[7:14, 10:-2] = (0, 0, 0, 0)
    seg[6, 11:-3]    = (0, 0, 0, 0)
    seg[5, 12:-4]    = (0, 0, 0, 0)
    seg[14, 11:-3]   = (0, 0, 0, 0)
    seg[15, 12:-4]   = (0, 0, 0, 0)
    seg[8:13, -2:]   = (0, 0, 0, 0)
# center the head
center(snake_head)
# style the body
with PixelArray(snake_body) as seg:
    # thin and barb sides
    seg[0, :]    = (0, 0, 0, 0)
    seg[1, ::2]  = (0, 0, 0, 0)
    seg[-2, ::2] = (0, 0, 0, 0)
    seg[-1:, :]  = (0, 0, 0, 0)
# center the body
center(snake_body)
# style the corner 1
with PixelArray(snake_cor1) as seg:
    # thin and barb outside
    seg[:, 0]    = (0, 0, 0, 0)
    seg[1::2, 1] = (0, 0, 0, 0)
    seg[1, ::2]  = (0, 0, 0, 0)
    seg[0, :]    = (0, 0, 0, 0)
    # inside in-cut
    seg[-1, -2:] = (0, 0, 0, 0)
    seg[-2, -2]  = (0, 0, 0, 0)
# center the corner 1
center(snake_cor1)
# style the corner 2
with PixelArray(snake_cor2) as seg:
    # thin and barb outside
    seg[:, 0]    = (0, 0, 0, 0)
    seg[::2, 1]  = (0, 0, 0, 0)
    seg[-2, ::2] = (0, 0, 0, 0)
    seg[-1, :]   = (0, 0, 0, 0)
    # inside in-cut
    seg[0, -2:] = (0, 0, 0, 0)
    seg[1, -2]  = (0, 0, 0, 0)
# center the corner 2
center(snake_cor2)
# style the tail
with PixelArray(snake_tail) as seg:
    # thin and barb sides
    seg[0, :]    = (0, 0, 0, 0)
    seg[1, ::2]  = (0, 0, 0, 0)
    seg[-2, ::2] = (0, 0, 0, 0)
    seg[-1:, :]  = (0, 0, 0, 0)
    # rounding
    seg[:, :2]    = (0, 0, 0, 0)
    seg[:4, 2:7]  = (0, 0, 0, 0)
    seg[1, 9]     = (0, 0, 0, 0)
    seg[:3, 7:9]  = (0, 0, 0, 0)
    seg[4, 2:5]   = (0, 0, 0, 0)
    seg[5, 2:4]   = (0, 0, 0, 0)
    seg[6, 2:3]   = (0, 0, 0, 0)
    seg[-4:, :7]  = (0, 0, 0, 0)
    seg[-2, 9]    = (0, 0, 0, 0)
    seg[-3:, 7:9] = (0, 0, 0, 0)
    seg[-5, 2:5]  = (0, 0, 0, 0)
    seg[-6, 2:4]  = (0, 0, 0, 0)
    seg[-7, 2:3]  = (0, 0, 0, 0)
# center the tail
center(snake_tail)

# win = Window(width=600, height=600, caption="Snake")
# spr = Sprite(snake_cor2, 290, 290)

# @win.event
# def on_draw():
#     win.clear()
#     spr.draw()

# pyglet.app.run()
# exit()


class Game:
    def __init__(self):
        self.event_loop = pyglet.app.EventLoop()
        self.clock = pyglet.clock.Clock()
        self.event_loop.clock = self.clock
        self.window = Window(600, 600, caption="Snake", vsync=False)
        self.main_batch = Batch()

        self.head = Sprite(snake_head, 290, 290, batch=self.main_batch)
        self.snake = [
            Sprite(snake_body, 270, 290, batch=self.main_batch),
            Sprite(snake_cor1, 250, 290, batch=self.main_batch),
            Sprite(snake_body, 250, 270, batch=self.main_batch),
        ]
        self.tail = Sprite(snake_tail, 250, 250, batch=self.main_batch)
        self.head.rotation = 90
        self.snake[0].rotation = 90
        self.snake[1].rotation = 90
        self.apple = Sprite(apple_img, 0, 0, batch=self.main_batch)
        self.gen_apple()  # ensures the apple doesn't overlap with the snake
        self.label = Label("Score: 0", x=10, y=580, batch=self.main_batch)
        self.window.event(self.on_draw)

        self.score = 0
        self.steps = 0
        self.turns = 0
        self.direction = 0
        self.exit_code = 0
        self.current_direction = 0
        self._fps = 0
        self._external = None
        self.fps = 3  # this also schedules the update method
        self.external = None  # this also adds the keyboard listeners

    def exit(self):
        self.event_loop.exit()
        self.window.close()

    def run(self):
        self.event_loop.run()

    def on_draw(self):
        self.window.clear()
        self.main_batch.draw()

    def gen_apple(self):
        valid = False
        while not valid:
            valid = True
            self.apple.update(x=random.randint(10, 590), y=random.randint(10, 590))
            if self.head.overlaps(self.apple):
                valid = False
                continue
            for s in self.snake:
                if s.overlaps(self.apple):
                    valid = False
                    break
            if valid:
                if self.tail.overlaps(self.apple):
                    valid = False

    def update(self, dt):
        self.steps += 1
        # if an external controller is hooked up, call it with the game instance
        if self._external:
            self._external(self)
        # forbid the head from running into the body
        if (self.current_direction - self.direction + 2) % 4 != 0:
            if self.current_direction != self.direction:
                self.turns += 1
            self.current_direction = self.direction
        else:
            # trying to go backwards
            forbidden_dirs = [0, 0, 0, 0]
            # snake body
            hx, hy = self.head.position
            for s in self.snake[2:]:
                if hx == s.x:
                    if hy + 20 == s.y:
                        forbidden_dirs[1] = 1
                    elif hy - 20 == s.y:
                        forbidden_dirs[3] = 1
                elif hy == s.y:
                    if hx + 20 == s.x:
                        forbidden_dirs[0] = 1
                    elif hx - 20 == s.x:
                        forbidden_dirs[2] = 1
            # out of bounds
            wx, wy = self.window.get_size()
            if hx < 20:
                forbidden_dirs[2] = 1
            elif hx > wx - 20:
                forbidden_dirs[0] = 1
            if hy < 20:
                forbidden_dirs[3] = 1
            elif hy > wy - 20:
                forbidden_dirs[1] = 1
            dirs = [i for i in (
                (self.direction - 1) % 4,
                (self.direction + 1) % 4
            ) if not forbidden_dirs[i]]
            if dirs:
                self.turns += 1
                self.current_direction = random.choice(dirs)
        # remember head position and rotation
        old_head, old_rot = Position(self.head.x, self.head.y), self.head.rotation
        # move and turn the head itself
        if self.current_direction == 0:
            self.head.move_and_turn(dx=20, rotation=90)
        elif self.current_direction == 1:
            self.head.move_and_turn(dy=20, rotation=0)
        elif self.current_direction == 2:
            self.head.move_and_turn(dx=-20, rotation=270)
        elif self.current_direction == 3:
            self.head.move_and_turn(dy=-20, rotation=180)
        # detect apple eating / collision
        if self.head.overlaps(self.apple):
            # score!
            self.score += 1
            self.steps = 0
            self.turns = 0
            self.label.text = f"Score: {self.score}"
            self.gen_apple()
        else:
            # remove the last segment
            # move the tail so that it's where the last segment is, with proper rotation
            segment = self.snake.pop()
            self.tail.update(x=segment.x, y=segment.y, rotation=segment.rotation)
        # create a new segment at old head location, adding it to the beginning of the list
        angle = (self.head.rotation - old_rot) % 360
        if angle % 180 == 0:
            segment = Sprite(snake_body, old_head.x, old_head.y, batch=self.main_batch)
            segment.rotation = self.head.rotation
        elif angle == 90:  # right turn
            segment = Sprite(snake_cor1, old_head.x, old_head.y, batch=self.main_batch)
            segment.rotation = self.head.rotation
        elif angle == 270:  # left turn
            segment = Sprite(snake_cor2, old_head.x, old_head.y, batch=self.main_batch)
            segment.rotation = self.head.rotation
        self.snake.insert(0, segment)  # type: ignore
        # detect collision with the snake body
        for segment in self.snake:
            if self.head.overlaps(segment):
                self.exit_code = 1
                self.event_loop.exit()
        # detect out-of-bounds movement
        if not self.head.overlaps(self.window, fully=True):
            self.exit_code = 2
            self.event_loop.exit()
        # detect stuck controller
        if self.steps > min(self.score * 100 + 200, 1000):
            self.exit_code = 3
            self.event_loop.exit()

    def reset(self):
        self.snake = self.snake[:3]
        self.head.update(x=290, y=290, rotation=90)
        self.tail.update(x=250, y=250, rotation=0)
        self.snake[0] = Sprite(snake_body, 270, 290, batch=self.main_batch)
        self.snake[1] = Sprite(snake_cor1, 250, 290, batch=self.main_batch)
        self.snake[2] = Sprite(snake_body, 250, 270, batch=self.main_batch)
        self.snake[0].rotation = 90
        self.snake[1].rotation = 90
        self.snake[2].rotation = 0
        self.gen_apple()
        self.label.text = "Score: 0"
        self.score = 0
        self.steps = 0
        self.turns = 0
        self.direction = 0
        self.exit_code = 0
        self.current_direction = 0

    def on_key_press(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.direction = 0
        elif symbol == key.UP:
            self.direction = 1
        elif symbol == key.LEFT:
            self.direction = 2
        elif symbol == key.DOWN:
            self.direction = 3
        elif symbol == key.SPACE:
            self.fps = 10

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.fps = 3

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps: int):
        self.clock.unschedule(self.update)
        if fps > 0:
            self.clock.schedule_interval(self.update, 1 / fps)
        self._fps = fps

    @property
    def external(self):
        return self._external

    @external.setter
    def external(self, func: Optional[Callable]):
        if func is None:
            self.window.event(self.on_key_press)
            self.window.event(self.on_key_release)
        else:
            self.window.remove_handler("on_key_press", self.on_key_press)
            self.window.remove_handler("on_key_release", self.on_key_release)
        self._external = func


if __name__ == "__main__":
    game = Game()
    while not game.window.has_exit:
        game.run()
        print(game.score)
        if game.window.has_exit:
            break
        sleep(2)
        game.reset()
