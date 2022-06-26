import numpy as np
import cv2

BACKGROUND_COLOR = (0, 0, 0)
SNAKE_BODY_COLOR = (0, 255, 0)
SNAKE_HEAD_COLOR = (0, 200, 100)
APPLE_COLOR = (0, 0, 255)


class DIRECTIONS:
    LEFT, UP, RIGHT, DOWN = range(4)


class ACTIONS:
    GO_LEFT, GO_UP, GO_RIGHT, GO_DOWN = range(4)


def coord_to_rectangle(x, y):
    return [(x, y), (x + 10, y + 10)]


class Apple():
    def __init__(self, BOUNDS: tuple[int, int], color: tuple[int, int, int] = APPLE_COLOR):
        self.bounds = BOUNDS
        self.color = color
        self.respawn()

    def respawn(self, x: int = None, y: int = None):
        x = np.random.randint((self.bounds[0] / 10) - 1) * 10 if x is None else x
        y = np.random.randint((self.bounds[1] / 10) - 1) * 10 if y is None else y
        self.position = (x, y)
        return self

    def draw(self, window) -> None:
        rectangle = coord_to_rectangle(self.position[0], self.position[1])
        cv2.rectangle(window, rectangle[0], rectangle[1], self.color, 3)


class Snake():
    def __init__(
        self,
        BOUNDS: tuple[int, int],
        body_color: tuple[int, int, int] = SNAKE_BODY_COLOR,
        head_color: tuple[int, int, int] = SNAKE_HEAD_COLOR
    ):
        self.bounds = BOUNDS
        self.body_color = body_color
        self.head_color = head_color
        self.is_dead = False
        self.respawn()

    def respawn(self):
        self.head = (int(self.bounds[0] / 2), int(self.bounds[1] / 2))  # spawn snake in center
        self.body = []
        self.direction = DIRECTIONS.LEFT
        self.is_dead = False
        return self

    def draw(self, window) -> None:
        for part in self.body:
            rectangle = coord_to_rectangle(part[0], part[1])
            cv2.rectangle(window, rectangle[0], rectangle[1], self.body_color, 3)
        rectangle = coord_to_rectangle(self.head[0], self.head[1])
        cv2.rectangle(window, rectangle[0], rectangle[1], self.head_color, 3)

    def will_die(self, new_head: int) -> bool:
        return (new_head[0] < 0
                or new_head[0] > self.bounds[0] - 1
                or new_head[1] < 0
                or new_head[1] > self.bounds[1] - 1
                or new_head in self.body)

    def length(self):
        return len(self.body) + 1

    def take_action(self, action, ate_apple) -> None:
        if action == ACTIONS.GO_LEFT and self.direction != DIRECTIONS.RIGHT:
            self.direction = DIRECTIONS.LEFT
        if action == ACTIONS.GO_UP and self.direction != DIRECTIONS.DOWN:
            self.direction = DIRECTIONS.UP
        if action == ACTIONS.GO_RIGHT and self.direction != DIRECTIONS.LEFT:
            self.direction = DIRECTIONS.RIGHT
        if action == ACTIONS.GO_DOWN and self.direction != DIRECTIONS.UP:
            self.direction = DIRECTIONS.DOWN

        new_head = self.head

        if self.direction == DIRECTIONS.LEFT:
            new_head = (new_head[0] - 10, new_head[1])
        elif self.direction == DIRECTIONS.UP:
            new_head = (new_head[0], new_head[1] - 10)
        elif self.direction == DIRECTIONS.RIGHT:
            new_head = (new_head[0] + 10, new_head[1])
        elif self.direction == DIRECTIONS.DOWN:
            new_head = (new_head[0], new_head[1] + 10)

        if self.will_die(new_head):
            self.is_dead = True
        else:
            self.body.append(self.head)
            self.head = new_head

            if not ate_apple:
                self.body.pop(0)


class Snake_Game():
    def __init__(self, title: str, bounds: tuple[int, int]):        
        self.title = title
        self.bounds: tuple[int, int] = bounds
        self.rendering: bool = False
        self.snake: Snake = None
        self.apple: Apple = None
        self.reset()

    def update(self, action):
        if self.ate_apple:
            self.score += 1
            self.apple.respawn()
        
        self.snake.take_action(action, self.ate_apple)
        self.ate_apple = self.snake.head == self.apple.position
        self.is_dead = self.snake.is_dead
        
    def reset(self):
        self.old_score = 0
        self.score = 0
        self.snake: Snake = self.snake.respawn() if self.snake is not None else Snake(self.bounds)
        self.apple: Apple = self.apple.respawn() if self.apple is not None else Apple(self.bounds)
        self.ate_apple = False
        self.is_dead = False
    
    def state(self) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int], bool, bool, int, int]:
        return self.snake.head, self.snake.body, self.apple.position, \
            self.ate_apple, self.is_dead, \
            self.snake.length(), self.score
    
    def draw(self, mode: str = 'human'):
        if mode == 'human':
            self.img = np.zeros((self.bounds[0], self.bounds[1], 3), dtype="uint8")
            if self.score > self.old_score:
                self.old_score = self.score
            else: 
                self.show_score()
                self.apple.draw(self.img)
                self.snake.draw(self.img)
            
            cv2.imshow(self.title, self.img)
            cv2.waitKey(1)         
            
        elif mode == 'print':
            print('Starting new game')
        
            if self.score > self.old_score:
                print(f"Found apple! Current score: {self.score}")

            if self.score < self.old_score:
                print(f"Finished game with score: {self.old_score}")

            self.old_score = self.score

    def show_score(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.img = np.zeros((self.bounds[0], self.bounds[1], 3), dtype='uint8')
        cv2.putText(
            self.img,
            f"SCORE: {self.score}", 
            (12, 12), 
            font, 0.5, (255, 255, 255), 1, 
            cv2.LINE_AA)
        
    def close(self):
        cv2.destroyAllWindows()
