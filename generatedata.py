#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成进阶Sokoban训练数据
特点：
1. 复杂地图（中间有随机墙）
2. 支持死锁检测和回溯
3. 支持瞬间移动（teleport）
4. 可配置各种动作的占比
"""

import json
import random
import argparse
from collections import deque
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from copy import deepcopy

# 方向定义
DIRS = {
    'U': (0, -1),
    'D': (0, 1),
    'L': (-1, 0),
    'R': (1, 0),
}

DIR_NAMES = {'U': '上', 'D': '下', 'L': '左', 'R': '右'}


@dataclass
class State:
    player: Tuple[int, int]
    boxes: frozenset

    def __hash__(self):
        return hash((self.player, self.boxes))

    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes


class Level:
    def __init__(self, width: int, height: int, walls: Set[Tuple[int, int]],
                 boxes: Set[Tuple[int, int]], goals: Set[Tuple[int, int]],
                 player: Tuple[int, int]):
        self.width = width
        self.height = height
        self.walls = walls
        self.boxes = boxes
        self.goals = goals
        self.player = player

    def is_floor(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return (x, y) not in self.walls

    def to_string(self, player_pos: Tuple[int, int], box_positions: Set[Tuple[int, int]]) -> str:
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos in self.walls:
                    row.append('#')
                elif pos == player_pos:
                    row.append('+' if pos in self.goals else '@')
                elif pos in box_positions:
                    row.append('*' if pos in self.goals else '$')
                elif pos in self.goals:
                    row.append('.')
                else:
                    row.append('_')
            lines.append(''.join(row))
        return '\n'.join(lines)


def get_reachable(level: Level, player: Tuple[int, int], boxes: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """获取玩家可达的所有位置"""
    visited = {player}
    queue = deque([player])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and level.is_floor(nx, ny) and (nx, ny) not in boxes:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


def is_corner_deadlock(level: Level, pos: Tuple[int, int], goals: Set[Tuple[int, int]]) -> bool:
    """检查位置是否是角落死锁"""
    if pos in goals:
        return False
    x, y = pos
    # 检查四个角
    up = not level.is_floor(x, y - 1)
    down = not level.is_floor(x, y + 1)
    left = not level.is_floor(x - 1, y)
    right = not level.is_floor(x + 1, y)
    return (up and left) or (up and right) or (down and left) or (down and right)


def is_deadlock(level: Level, boxes: Set[Tuple[int, int]]) -> bool:
    """检查当前状态是否死锁"""
    for box in boxes:
        if box not in level.goals and is_corner_deadlock(level, box, level.goals):
            return True
    return False


def can_push(level: Level, player: Tuple[int, int], boxes: Set[Tuple[int, int]], direction: str) -> Tuple[bool, str]:
    dx, dy = DIRS[direction]
    front = (player[0] + dx, player[1] + dy)

    if not level.is_floor(front[0], front[1]):
        return False, "墙"

    if front in boxes:
        box_dest = (front[0] + dx, front[1] + dy)
        if not level.is_floor(box_dest[0], box_dest[1]):
            return False, "箱子后是墙"
        if box_dest in boxes:
            return False, "箱子后是箱子"
        return True, "推箱子"

    return True, "移动"


def do_move(player: Tuple[int, int], boxes: Set[Tuple[int, int]], direction: str) -> Tuple[Tuple[int, int], Set[Tuple[int, int]]]:
    dx, dy = DIRS[direction]
    front = (player[0] + dx, player[1] + dy)
    new_boxes = set(boxes)
    if front in boxes:
        new_boxes.remove(front)
        new_boxes.add((front[0] + dx, front[1] + dy))
    return front, new_boxes


def bfs_solve(level: Level, max_steps: int = 100) -> Optional[List[str]]:
    """BFS求解"""
    initial = State(level.player, frozenset(level.boxes))
    if initial.boxes == frozenset(level.goals):
        return []

    queue = deque([(initial, [])])
    visited = {initial}

    while queue:
        state, path = queue.popleft()
        if len(path) >= max_steps:
            continue

        for direction in ['U', 'D', 'L', 'R']:
            can, _ = can_push(level, state.player, set(state.boxes), direction)
            if can:
                new_player, new_boxes = do_move(state.player, set(state.boxes), direction)
                new_state = State(new_player, frozenset(new_boxes))

                if new_state not in visited:
                    visited.add(new_state)
                    new_path = path + [direction]

                    if new_state.boxes == frozenset(level.goals):
                        return new_path

                    # 跳过死锁状态
                    if not is_deadlock(level, set(new_state.boxes)):
                        queue.append((new_state, new_path))

    return None


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def min_distance_to_goals(pos: Tuple[int, int], goals: Set[Tuple[int, int]]) -> int:
    if not goals:
        return 0
    return min(manhattan(pos, g) for g in goals)


def has_push_space(level: Level, pos: Tuple[int, int], boxes: Set[Tuple[int, int]]) -> bool:
    x, y = pos
    for dx, dy in DIRS.values():
        front = (x + dx, y + dy)
        back = (x - dx, y - dy)
        if level.is_floor(front[0], front[1]) and level.is_floor(back[0], back[1]):
            if front not in boxes and back not in boxes:
                return True
    return False


def generate_complex_level(rng: random.Random, num_boxes: int, width: int, height: int,
                           inner_walls: int = 0, min_goal_dist: int = 0) -> Optional[Level]:
    """生成带内部墙的复杂关卡"""
    # 创建边界墙
    walls = set()
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    # 内部空间
    interior = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1)]

    # 添加内部墙（随机位置）
    if inner_walls > 0 and len(interior) > num_boxes * 3 + inner_walls + 2:
        rng.shuffle(interior)
        for i in range(min(inner_walls, len(interior) // 4)):
            wall_pos = interior.pop()
            walls.add(wall_pos)

    # 重新获取可用内部空间
    interior = [p for p in interior if p not in walls]

    if len(interior) < num_boxes * 2 + 2:
        return None

    # 检查连通性
    def is_connected(floors):
        if not floors:
            return False
        start = next(iter(floors))
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS.values():
                nx, ny = x + dx, y + dy
                if (nx, ny) in floors and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited) == len(floors)

    if not is_connected(set(interior)):
        return None

    rng.shuffle(interior)

    # 选择目标和箱子位置
    goals = set()
    boxes = set()
    used = set()

    for _ in range(num_boxes):
        placed = False
        for pos in interior:
            if pos in used:
                continue
            x, y = pos
            # 确保不是角落死锁位置
            has_space = False
            for dx, dy in DIRS.values():
                adj = (x + dx, y + dy)
                opp = (x - dx, y - dy)
                if adj in interior and adj not in used and opp in interior and opp not in used:
                    has_space = True
                    break
            if has_space:
                goals.add(pos)
                used.add(pos)
                placed = True
                break
        if not placed:
            return None

    level = Level(width, height, walls, set(), goals, (0, 0))

    # 放置箱子（不强制贴近目标，增加解法多样性）
    candidates = [p for p in interior if p not in goals]
    rng.shuffle(candidates)
    for pos in candidates:
        if len(boxes) >= num_boxes:
            break
        if pos in used:
            continue
        if min_goal_dist > 0 and min_distance_to_goals(pos, goals) < min_goal_dist:
            continue
        if is_corner_deadlock(level, pos, goals):
            continue
        if not has_push_space(level, pos, boxes):
            continue
        boxes.add(pos)
        used.add(pos)

    if len(boxes) < num_boxes:
        return None

    # 选择玩家位置
    player_candidates = [p for p in interior if p not in used and p not in goals and p not in boxes]
    if not player_candidates:
        return None

    player = rng.choice(player_candidates)

    return Level(width, height, walls, boxes, goals, player)


def generate_cot_simple(level: Level, solution: List[str]) -> str:
    """生成简单CoT（无回溯）"""
    parts = []
    player = level.player
    boxes = set(level.boxes)

    for direction in solution:
        state_str = level.to_string(player, boxes)
        parts.append(f"<state>\n{state_str}\n</state>")

        dx, dy = DIRS[direction]
        front = (player[0] + dx, player[1] + dy)
        is_push = front in boxes

        dir_cn = DIR_NAMES[direction]
        if is_push:
            box_dest = (front[0] + dx, front[1] + dy)
            if box_dest in level.goals:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子到目标</action>")
            else:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子</action>")
        else:
            parts.append(f"<action>{direction}:向{dir_cn}移动</action>")

        player, boxes = do_move(player, boxes, direction)

    final_state = level.to_string(player, boxes)
    parts.append(f"<done>\n{final_state}\n</done>")

    return '\n'.join(parts)


def generate_cot_with_backtrack(level: Level, solution: List[str], rng: random.Random,
                                 max_backtracks: int = 2) -> str:
    """生成带回溯的CoT"""
    parts = []
    player = level.player
    boxes = set(level.boxes)

    # 记录历史状态用于回溯
    history = [(player, set(boxes))]

    i = 0
    backtracks_done = 0

    while i < len(solution):
        state_str = level.to_string(player, boxes)
        parts.append(f"<state>\n{state_str}\n</state>")

        # 随机决定是否尝试错误动作（导致死锁）
        if backtracks_done < max_backtracks and rng.random() < 0.15:
            # 尝试一个可能导致死锁的动作
            wrong_dirs = [d for d in ['U', 'D', 'L', 'R'] if d != solution[i]]
            rng.shuffle(wrong_dirs)

            tried_wrong = False
            for wrong_dir in wrong_dirs:
                can, _ = can_push(level, player, boxes, wrong_dir)
                if can:
                    test_player, test_boxes = do_move(player, boxes, wrong_dir)
                    if is_deadlock(level, test_boxes):
                        # 记录错误尝试
                        dir_cn = DIR_NAMES[wrong_dir]
                        dx, dy = DIRS[wrong_dir]
                        front = (player[0] + dx, player[1] + dy)
                        if front in boxes:
                            parts.append(f"<action>{wrong_dir}:向{dir_cn}推箱子</action>")
                        else:
                            parts.append(f"<action>{wrong_dir}:向{dir_cn}移动</action>")

                        # 死锁状态
                        dead_state = level.to_string(test_player, test_boxes)
                        parts.append(f"<dead>\n{dead_state}\n</dead>")

                        # 回溯
                        parts.append(f"<back>回退</back>")
                        backtracks_done += 1
                        tried_wrong = True
                        break

            if tried_wrong:
                # 重新显示当前状态，继续正确路径
                continue

        # 执行正确动作
        direction = solution[i]
        dx, dy = DIRS[direction]
        front = (player[0] + dx, player[1] + dy)
        is_push = front in boxes

        dir_cn = DIR_NAMES[direction]
        if is_push:
            box_dest = (front[0] + dx, front[1] + dy)
            if box_dest in level.goals:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子到目标</action>")
            else:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子</action>")
        else:
            parts.append(f"<action>{direction}:向{dir_cn}移动</action>")

        player, boxes = do_move(player, boxes, direction)
        history.append((player, set(boxes)))
        i += 1

    final_state = level.to_string(player, boxes)
    parts.append(f"<done>\n{final_state}\n</done>")

    return '\n'.join(parts)


def generate_cot_with_backtrack_prefill(level: Level, solution: List[str], rng: random.Random,
                                        max_prefix_steps: int = 10,
                                        max_wrong_len: int = 2) -> Optional[Tuple[str, str]]:
    """生成带回溯的CoT（错误动作作为预填充）"""
    if not solution:
        return None

    def find_deadlock_seq(player, boxes, correct_dir):
        wrong_len = 2 if (max_wrong_len >= 2 and rng.random() < 0.4) else 1
        for length in ([2, 1] if wrong_len == 2 else [1]):
            candidates = [(a, b) for a in DIRS for b in DIRS] if length == 2 else [(a,) for a in DIRS]
            rng.shuffle(candidates)
            for seq in candidates:
                if seq[0] == correct_dir:
                    continue
                p = player
                bxs = set(boxes)
                ok = True
                for d in seq:
                    can, _ = can_push(level, p, bxs, d)
                    if not can:
                        ok = False
                        break
                    p, bxs = do_move(p, bxs, d)
                if not ok:
                    continue
                if is_deadlock(level, bxs):
                    return list(seq), p, bxs
        return None

    prefill_parts = []
    player = level.player
    boxes = set(level.boxes)

    for i, direction in enumerate(solution[:max_prefix_steps]):
        state_str = level.to_string(player, boxes)
        deadlock_result = find_deadlock_seq(player, boxes, direction)
        if deadlock_result:
            wrong_seq, dead_player, dead_boxes = deadlock_result
            prefill_parts.append(f"<state>\n{state_str}\n</state>")
            for wrong_dir in wrong_seq:
                dx, dy = DIRS[wrong_dir]
                front = (player[0] + dx, player[1] + dy)
                is_push = front in boxes
                dir_cn = DIR_NAMES[wrong_dir]
                if is_push:
                    box_dest = (front[0] + dx, front[1] + dy)
                    if box_dest in level.goals:
                        prefill_parts.append(f"<action>{wrong_dir}:向{dir_cn}推箱子到目标</action>")
                    else:
                        prefill_parts.append(f"<action>{wrong_dir}:向{dir_cn}推箱子</action>")
                else:
                    prefill_parts.append(f"<action>{wrong_dir}:向{dir_cn}移动</action>")

                player, boxes = do_move(player, boxes, wrong_dir)

            parts = [f"<dead>\n{level.to_string(dead_player, dead_boxes)}\n</dead>"]
            for _ in range(len(wrong_seq)):
                parts.append("<back>回退</back>")

            player = level.player
            boxes = set(level.boxes)
            for direction in solution[:i]:
                player, boxes = do_move(player, boxes, direction)

            for direction in solution[i:]:
                state_str = level.to_string(player, boxes)
                parts.append(f"<state>\n{state_str}\n</state>")

                dx, dy = DIRS[direction]
                front = (player[0] + dx, player[1] + dy)
                is_push = front in boxes

                dir_cn = DIR_NAMES[direction]
                if is_push:
                    box_dest = (front[0] + dx, front[1] + dy)
                    if box_dest in level.goals:
                        parts.append(f"<action>{direction}:向{dir_cn}推箱子到目标</action>")
                    else:
                        parts.append(f"<action>{direction}:向{dir_cn}推箱子</action>")
                else:
                    parts.append(f"<action>{direction}:向{dir_cn}移动</action>")

                player, boxes = do_move(player, boxes, direction)

            final_state = level.to_string(player, boxes)
            parts.append(f"<done>\n{final_state}\n</done>")

            return '\n'.join(prefill_parts), '\n'.join(parts)

        prefill_parts.append(f"<state>\n{state_str}\n</state>")
        dx, dy = DIRS[direction]
        front = (player[0] + dx, player[1] + dy)
        is_push = front in boxes
        dir_cn = DIR_NAMES[direction]
        if is_push:
            box_dest = (front[0] + dx, front[1] + dy)
            if box_dest in level.goals:
                prefill_parts.append(f"<action>{direction}:向{dir_cn}推箱子到目标</action>")
            else:
                prefill_parts.append(f"<action>{direction}:向{dir_cn}推箱子</action>")
        else:
            prefill_parts.append(f"<action>{direction}:向{dir_cn}移动</action>")
        player, boxes = do_move(player, boxes, direction)

    return None


def generate_cot_with_teleport(level: Level, solution: List[str], rng: random.Random) -> str:
    """生成带瞬间移动的CoT"""
    parts = []
    player = level.player
    boxes = set(level.boxes)

    i = 0
    skip_tele_until = -1
    while i < len(solution):
        state_str = level.to_string(player, boxes)
        parts.append(f"<state>\n{state_str}\n</state>")

        if i >= skip_tele_until:
            move_seq = []
            temp_player = player
            temp_boxes = set(boxes)
            j = i
            while j < len(solution):
                direction = solution[j]
                next_player, next_boxes = do_move(temp_player, temp_boxes, direction)
                if next_boxes != temp_boxes:
                    break
                move_seq.append(direction)
                temp_player, temp_boxes = next_player, next_boxes
                j += 1

            if len(move_seq) > 5:
                skip_tele_until = i + len(move_seq)
            elif 3 <= len(move_seq) <= 5:
                parts.append(f"<tele>传送到({temp_player[0]},{temp_player[1]})</tele>")
                player = temp_player
                i += len(move_seq)
                continue

        # 正常执行一步
        direction = solution[i]
        dx, dy = DIRS[direction]
        front = (player[0] + dx, player[1] + dy)
        is_push = front in boxes

        dir_cn = DIR_NAMES[direction]
        if is_push:
            box_dest = (front[0] + dx, front[1] + dy)
            if box_dest in level.goals:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子到目标</action>")
            else:
                parts.append(f"<action>{direction}:向{dir_cn}推箱子</action>")
        else:
            parts.append(f"<action>{direction}:向{dir_cn}移动</action>")

        player, boxes = do_move(player, boxes, direction)
        i += 1

    final_state = level.to_string(player, boxes)
    parts.append(f"<done>\n{final_state}\n</done>")

    return '\n'.join(parts)


def generate_dataset(num_samples: int, num_boxes: int,
                     width_range: Tuple[int, int], height_range: Tuple[int, int],
                     seed: int, max_solution_len: int,
                     inner_walls_range: Tuple[int, int] = (0, 0),
                     backtrack_ratio: float = 0.0,
                     teleport_ratio: float = 0.0,
                     min_solution_len: int = 0,
                     min_goal_dist: int = 0,
                     backtrack_prefill: bool = False,
                     require_backtrack: bool = False,
                     require_teleport: bool = False,
                     output_path: Optional[str] = None,
                     flush_every: int = 1):
    """生成数据集"""
    rng = random.Random(seed)
    dataset = [] if output_path is None else None
    attempts = 0
    max_attempts = num_samples * (400 if (require_backtrack or require_teleport) else 200)
    total_len = 0
    cot_types: Dict[str, int] = {}
    sample = None

    writer = None
    if output_path:
        writer = open(output_path, 'w', encoding='utf-8')

    count = 0
    while count < num_samples and attempts < max_attempts:
        attempts += 1

        width = rng.randint(width_range[0], width_range[1])
        height = rng.randint(height_range[0], height_range[1])
        inner_walls = rng.randint(inner_walls_range[0], inner_walls_range[1])

        level = generate_complex_level(
            rng, num_boxes, width, height,
            inner_walls=inner_walls,
            min_goal_dist=min_goal_dist,
        )
        if level is None:
            continue

        solution = bfs_solve(level, max_steps=max_solution_len)
        if solution is None or len(solution) == 0 or len(solution) < min_solution_len:
            continue

        # 决定CoT类型
        roll = rng.random()
        prefill = ""
        if roll < backtrack_ratio:
            if backtrack_prefill:
                prefill_result = generate_cot_with_backtrack_prefill(level, solution, rng)
                if prefill_result is None:
                    continue
                prefill, cot_str = prefill_result
            else:
                cot_str = generate_cot_with_backtrack(level, solution, rng)
            cot_type = "backtrack"
        elif roll < backtrack_ratio + teleport_ratio:
            cot_str = generate_cot_with_teleport(level, solution, rng)
            cot_type = "teleport"
        else:
            cot_str = generate_cot_simple(level, solution)
            cot_type = "simple"

        if cot_type == "teleport" and require_teleport and "<tele>" not in cot_str:
            continue
        if cot_type == "backtrack" and require_backtrack and not prefill:
            continue

        input_str = f"<input>\n{level.to_string(level.player, level.boxes)}\n</input>"
        if prefill:
            input_str = input_str + "\n" + prefill

        item = {
            "input": input_str,
            "cot": cot_str,
            "solution_length": len(solution),
            "num_boxes": num_boxes,
            "width": width,
            "height": height,
            "inner_walls": inner_walls,
            "cot_type": cot_type,
            "solution": ''.join(solution),
        }

        if dataset is not None:
            dataset.append(item)
        else:
            writer.write(json.dumps(item, ensure_ascii=False) + '\n')
            if count % max(1, flush_every) == 0:
                writer.flush()

        if sample is None:
            sample = item

        count += 1
        total_len += len(solution)
        cot_types[cot_type] = cot_types.get(cot_type, 0) + 1

        if count % 500 == 0:
            print(f"Generated {count} samples...")

    if writer:
        writer.flush()
        writer.close()

    return dataset, count, total_len, cot_types, sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_boxes", type=int, default=2)
    parser.add_argument("--width_min", type=int, default=6)
    parser.add_argument("--width_max", type=int, default=8)
    parser.add_argument("--height_min", type=int, default=6)
    parser.add_argument("--height_max", type=int, default=8)
    parser.add_argument("--inner_walls_min", type=int, default=0)
    parser.add_argument("--inner_walls_max", type=int, default=3)
    parser.add_argument("--max_solution_len", type=int, default=25)
    parser.add_argument("--min_solution_len", type=int, default=4)
    parser.add_argument("--min_goal_dist", type=int, default=1)
    parser.add_argument("--backtrack_ratio", type=float, default=0.0)
    parser.add_argument("--teleport_ratio", type=float, default=0.0)
    parser.add_argument("--backtrack_prefill", action="store_true")
    parser.add_argument("--require_backtrack", action="store_true")
    parser.add_argument("--require_teleport", action="store_true")
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="advanced_train.jsonl")

    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples with {args.num_boxes} boxes...")
    print(f"Map size: {args.width_min}-{args.width_max} x {args.height_min}-{args.height_max}")
    print(f"Inner walls: {args.inner_walls_min}-{args.inner_walls_max}")
    print(f"Min solution length: {args.min_solution_len}")
    print(f"Min box-goal distance: {args.min_goal_dist}")
    print(f"Backtrack ratio: {args.backtrack_ratio}, Teleport ratio: {args.teleport_ratio}")

    dataset, count, total_len, cot_types, sample = generate_dataset(
        num_samples=args.num_samples,
        num_boxes=args.num_boxes,
        width_range=(args.width_min, args.width_max),
        height_range=(args.height_min, args.height_max),
        seed=args.seed,
        max_solution_len=args.max_solution_len,
        inner_walls_range=(args.inner_walls_min, args.inner_walls_max),
        backtrack_ratio=args.backtrack_ratio,
        teleport_ratio=args.teleport_ratio,
        min_solution_len=args.min_solution_len,
        min_goal_dist=args.min_goal_dist,
        backtrack_prefill=args.backtrack_prefill,
        require_backtrack=args.require_backtrack,
        require_teleport=args.require_teleport,
        output_path=args.output,
        flush_every=args.flush_every,
    )

    print(f"\nDone! Generated {count} samples to {args.output}")

    if count > 0:
        avg_len = total_len / count
        print(f"Average solution length: {avg_len:.2f}")
        print(f"CoT types: {cot_types}")

        if sample:
            print("\n" + "="*60)
            print("Sample example:")
            print("="*60)
            print(f"Input:\n{sample['input']}")
            print(f"\nCoT:\n{sample['cot'][:800]}...")


if __name__ == "__main__":
    main()
