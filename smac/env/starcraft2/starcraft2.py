from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import ValuesView

from smac.env.multiagentenv import MultiAgentEnv
from smac.env.starcraft2.maps import get_map_params
# get_map_params 함수는 maps 폴더의 __init__.py에 정의 되어있으며 같은폴더내의 smac_maps.py에 있는 map_param_registry 를 읽어온다
# map_param_registry (맵 파라미터 레지스트)는 Starcraft2의 실행파일이 있는 폴더에 있는 SMAC_Maps (실제 제작된 지도들)에 있는 맵정보에 축약된 정보를 담고 있으며,
# 아무래도 맵에 있는 정보를 직접적으로 읽어와서 매개변수로 던져놓는게 안되다보니 메뉴얼하게 저장된 맵정보를 저장하여 이 정보를 매개변수로 던져주는 듯하다.


print('#######################proper starcraft version#####################')

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

## Refil
#from numpy.random import RandomState


# sc2 api가 제공 map 정보 제공
from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib import remote_controller as py_rc

# Refil
#from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
#from pysc2.lib.units import get_unit_type



from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

# 종족에 관한 정보를 담고 있으며, R,P,T,Z 로 인코딩된 문자는 각각 랜덤 플토 테란 저그를 의미


difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}

# 난이도 관련 설정을 담당하며 대부분의 게임에서 7로 설정함(상대방 휴리스틱 AI의 난이도를 결정하는 숫자임)


actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
    "morph_siegemode" : 2989,
    #"morph_siegemode" : 388,
    #"morph_unsiege" : 390
    "morph_unsiege" : 2991
}

# 딕셔너리 자료로 액션이름과 인코딩된 값들을 키밸류형태로 저장함




# Refil @@@@

def get_unit_name_by_type(utype):
    if utype == 1935:
        return 'Baneling_RL'
    elif utype == 9:
        return 'Baneling'
    elif utype == 1936:
        return 'Colossus_RL'
    elif utype == 4:
        return 'Colossus'
    elif utype == 1937:
        return 'Hydralisk_RL'
    elif utype == 107:
        return 'Hydralisk'
    elif utype == 1938:
        return 'Marauder_RL'
    elif utype == 51:
        return 'Marauder'
    elif utype == 1939:
        return 'Marine_RL'
    elif utype == 48:
        return 'Marine'
    elif utype == 1940:
        return 'Medivac_RL'
    elif utype == 54:
        return 'Medivac'
    elif utype == 1941:
        return 'Stalker_RL'
    elif utype == 74:
        return 'Stalker'
    elif utype == 1942:
        return 'Zealot_RL'
    elif utype == 73:
        return 'Zealot'
    elif utype == 1943:
        return 'Zergling_RL'
    elif utype == 105:
        return 'Zergling'
    elif utype == 3220:
        return 'Ghost_RL'
    elif utype == 50:
        return 'Ghost'
    elif utype == 919:
        return 'Siege_Tank_RL_Tankmode'
    elif utype == 2178:
        return 'Siege_Tank_Tankmode'
    elif utype == 918:
        return 'Siege_Tank_RL_Siegemode'
    elif utype == 2177:
        return 'Siege_Tank_Siegemode'
    # Below is Neutral, Terrain Entity
    elif utype == 1235:
        return 'Tree'
    elif utype == 1236:
        return 'Bush'
    elif utype == 312:
        return 'Rocks_2by2'
    elif utype == 1763:
        return 'Rocks_H2by4'
    elif utype == 1762:
        return 'Rocks_V4by2'


def get_unit_type_by_name(name, custom=False):
    """
    If custom, return special *_RL unit type id
    These units turn off any automated return fire so they are controlled
    precisely by the RL agents (use for ally units)
    """
    if custom:
        if name == 'Baneling':
            return 1935
        elif name == 'Colossus':
            return 1936
        elif name == 'Hydralisk':
            return 1937
        elif name == 'Marauder':
            return 1938
        elif name == 'Marine':
            return 1939
        elif name == 'Medivac':
            return 1940
        elif name == 'Stalker':
            return 1941
        elif name == 'Zealot':
            return 1942
        elif name == 'Zergling':
            return 1943
        # These below are newly added
        elif name == 'Ghost_RL':
            return 3220
        elif name == 'Ghost':
            return 50
        elif name == 'Siege_Tank_RL_Tankmode':
            return 919
        elif name == 'Siege_Tank_Tankmode':
            return 2178
        elif name == 'Siege_Tank_RL_Siegemode':
            return 918
        elif name == 'Siege_Tank_Siegemode':
            return 2177
        # Below is Neutral, Terrain Entity
        elif name == 'Tree':
            return 1235
        elif name == 'Bush':
            return 1236
        elif name == 'Rocks_2by2':
            return 312
        elif name == 'Rocks_H2by4':
            return 1763
        elif name == 'Rocks_V4by2':
            return 1762
    else:
        for race in (Neutral, Protoss, Terran, Zerg):
            unit = getattr(race, name, None)
            if unit is not None:
                return unit
        raise ValueError("Bad unit type {}".format(name))
# Refil @@@@










class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

# 동서남북으로 방향값들을 정하는듯, enum.IntEnum 은 파이썬 기본모듈로 제공됨


# StarCraft2Env는 MultiAgentEnv 를 상속받는다. MultiAgentEnv는 Pymarl에서 env 폴더의 __init__ 모듈에 의해 레지스트리형태로 담긴 정보가 episode_runner로 넘어가고
# episode_runner(또는 파랄렐)해당 정보를 토대로 호출된 뒤 이를 호출하는 run함수가 돌아갈때 사용된다.
class StarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(
        self,
        map_name="8m",
        step_mul=8,
        move_amount=2,
        difficulty="7",
        game_version=None,
        seed=None,
        continuing_episode=False,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=False,
         # pathing_grid and terrain_height info is changed to be given by default
        obs_pathing_grid=True,
        obs_terrain_height=True,
        obs_instead_of_state=False,
        obs_timestep_number=False,
        obs_broadcast_info=True,
        obs_communicate_info=True,
        state_last_action=True,
        state_timestep_number=False,
        reward_sparse=False,
        reward_only_positive=True,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        replay_dir="",
        replay_prefix="",
        window_size_x=1920,
        window_size_y=1200,
        heuristic_ai=False,
        heuristic_rest=False,
        debug=False,
    ):
        # 맵이름은 8마린(문자열, 옵션), step_mul(정수, 옵션)은 에이전트의 스텝당 게임스텝이다.(pysc2랑 연관), move_amount(실수, 옵션)는 뒤에 나올 이동액션을 취해졌을때 x,y 좌푯값으로 이동하는 이동량이다
        # 난이도(문자열, 옵션)는 빌트인AI의 난이도로 일반적으로 7로 설정한다.
        # 게임버전(문자열, 옵션)은 전체 코드에서 None으로 되어있는데 이는 최신버전을 의미한다, pysc2 쪽에서 리플레이 돌릴때는 관련 정보가 있다. seed값(정수, 옵션)도 launch함수가 실행될때 랜덤 샘플링된 값이 넘어간다.
        # continuing_episode (부울, 옵션)는 맵의 타임리밋이 되었을때 에피소드가 진행중인것으로 볼지 끝난것으로 볼지인데 기본값은 거짓으로 되어있다(끝난것으로 본다는 의미겠지??)
        # obs_all_health (부울, 옵션)는 에이전트가 시야에 있는 모든 유닛의 체력 정보를 관측으로 획득하는지 여부인데, 기본은 참으로 되어있어 모두의(시야안) 체력정보를 획득한다
        # obs_own_health (부울, 옵션) 각 에이전트가 그들의 자신의 체력을 관측으로 받을지 여부인데 기본이 거짓이므로 정보를 받지않는다. 하지만 바로 위에 있는 obs_all_health가 트루이면 비활성화(무시)되는 옵션이므로 결론적으로는 에이전트는 자기포함 전유닛의 체력정보를 받는다.
        # obs_last_action (부울, 옵션) 에이전트가 시야내의 모든 유닛의 마지막 행동정보를 수집하는지 여부이다. 기본은 거짓이나 이는 training과 eval 일때가 다를것 같다(확인해보겟음)
        # obs_pathing_grid : (부울, 옵션) 기본은 거짓으로 설정되어있음. 관측정보에 에이전트 근처에 pathing value를 포함할지를 결정하는 값이다
        # obs_terrain_height : (부울, 옵션) 에이전트 관측정보에 지형높이값이 저장될지 여부, 기본은 거짓임
        # obs_instead_of_state : (부울, 옵션) 모든 에이전트의 관측의 조합으로 글로벌 상태정보를 대체할지 여부인데 기본은 거짓으로 별도로 정보를 넣어줌
        # obs_timestep_number : (부울, 옵션) 현재 에피소드의 타임스텝 정보를 관측에 넣어줄지 여부(기본은 거짓)
        # reward_sparse : (부울, 옵션) 이기거나 질때 1/-1 의 리워드를 부여할지 여부로 기본은 거짓임 (참, sparse하게 부여시 다른 기존의 reward는 부여되지않음)
        # reward_only_positive : (부울, 옵션) 보상이 오직 긍정적일지 여부 (보상은 오직 긍정적인 거만 있음, 기본이 참임)
        # reward_death_value : (실수, 옵션) 적군 유닛 죽였을때 받는보상의 양을 이걸로 정할수 있음. reward_only_positive가 거짓이면 reward_death_value에 의해 정해진 값만큼 아군 에이전트 죽을때 보상으로 받음
        # reward_win : (실수, 옵션) 승리했을시 보상값 결정 (기본 200)
        # reward_defeat : (실수, 옵션) 패배했을시 보상값 결정 (기본 0), non-positive 여야함(양의 값이면 에러날것같음)
        # reward_negative_scale : (실수, 옵션)  음의 보상을 받을때의 스케일링 팩터임 reward_only_positive가 True(해당파라미터의 기본값)로 설정되면 무시됨, 기본값을 0.5임
        # reward_scale : (부울, 옵션) 보상에 스케일링을 할지말지를 결정하는 인자로 기본은 참임
        # reward_scale_rate : (실수, 옵션) 기본값은 20이며, reward_scale 이 참으로 되어있으면, 에이전트가 받는 보상은 프로토스유닛의 쉴드 재생을 고려하지않고 계산된
        # 에피소드당 최대가능 보상(가능한 최대치)에서 reward_scale_rate 만큼 나누어서 받음
        # reward per episode without considering the shield regeneration of Protoss units.
        # replay_dir : (문자열, 옵션) 리플레이저장하는 디렉토리로 기본은 None인데 이렇게 설정되면 설치된 StarCraft2 폴더에 리플레이 디렉토리에 저장된다

        # replay_prefix : (문자열, 옵션) 저장될 리플레이의 접두사로 기본은 None인데 이렇게 되어있으면 지도이름이 사용되어 저장된다.

        # window_size_x : (정수, 옵션) SC2 윈도우 가로 크기로 기본은 1920 이다
        # window_size_y: (정수, 옵션)  SC2 윈도우 세로 크기로 기본은 1200 이다
        # heuristic_ai: (부울, 옵션) 학습하지 않는 휴리스틱 AI를 쓸지말지 여부로 기본은 거짓으로 되어있다
        # heuristic_rest: (부울, 옵션) 아무때나 휴리스틱 AI의 행동을 RL에이전트에게 가용한 행동으로부터 선택하여 제약을 가할 수 있는지 여부, 기본은 거짓이다, 휴리스틱 AI가 거짓으로 되어있으면 무시된다.
        # 요건 좀 이해가 안됨~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~@##&$&^#ㅒㅑ#&&*@&
        # debug: (부울, 옵션) 관측과 상태, 행동, 보상에 대한 메시지를 디버깅 목적으로 로깅할지 여부이다.(기본이 거짓임)
        """
        Create a StarCraftC2Env environment.

        Parameters
        ----------
        map_name : str, optional
            The name of the SC2 map to play (default is "8m"). The full list
            can be found by running bin/map_list.
        step_mul : int, optional
            How many game steps per agent step (default is 8). None
            indicates to use the default map step_mul.
        move_amount : float, optional
            How far away units are ordered to move per step (default is 2).
        difficulty : str, optional
            The difficulty of built-in computer AI bot (default is "7").
        game_version : str, optional
            StarCraft II game version (default is None). None indicates the
            latest version.
        seed : int, optional
            Random seed used during game initialisation. This allows to
        continuing_episode : bool, optional
            Whether to consider episodes continuing or finished after time
            limit is reached (default is False).
        obs_all_health : bool, optional
            Agents receive the health of all units (in the sight range) as part
            of observations (default is True).
        obs_own_health : bool, optional
            Agents receive their own health as a part of observations (default
            is False). This flag is ignored when obs_all_health == True.
        obs_last_action : bool, optional
            Agents receive the last actions of all units (in the sight range)
            as part of observations (default is False).
        obs_pathing_grid : bool, optional
            Whether observations include pathing values surrounding the agent
            (default is False).
        obs_terrain_height : bool, optional
            Whether observations include terrain height values surrounding the
            agent (default is False).
        obs_instead_of_state : bool, optional
            Use combination of all agents' observations as the global state
            (default is False).
        obs_timestep_number : bool, optional
            Whether observations include the current timestep of the episode
            (default is False).
        state_last_action : bool, optional
            Include the last actions of all agents as part of the global state
            (default is True).
        state_timestep_number : bool, optional
            Whether the state include the current timestep of the episode
            (default is False).
        reward_sparse : bool, optional
            Receive 1/-1 reward for winning/loosing an episode (default is
            False). Whe rest of reward parameters are ignored if True.
        reward_only_positive : bool, optional
            Reward is always positive (default is True).
        reward_death_value : float, optional
            The amount of reward received for killing an enemy unit (default
            is 10). This is also the negative penalty for having an allied unit
            killed if reward_only_positive == False.
        reward_win : float, optional
            The reward for winning in an episode (default is 200).
        reward_defeat : float, optional
            The reward for loosing in an episode (default is 0). This value
            should be nonpositive.
        reward_negative_scale : float, optional
            Scaling factor for negative rewards (default is 0.5). This
            parameter is ignored when reward_only_positive == True.
        reward_scale : bool, optional
            Whether or not to scale the reward (default is True).
        reward_scale_rate : float, optional
            Reward scale rate (default is 20). When reward_scale == True, the
            reward received by the agents is divided by (max_reward /
            reward_scale_rate), where max_reward is the maximum possible
            reward per episode without considering the shield regeneration
            of Protoss units.
        replay_dir : str, optional
            The directory to save replays (default is None). If None, the
            replay will be saved in Replays directory where StarCraft II is
            installed.
        replay_prefix : str, optional
            The prefix of the replay to be saved (default is None). If None,
            the name of the map will be used.
        window_size_x : int, optional
            The length of StarCraft II window size (default is 1920).
        window_size_y: int, optional
            The height of StarCraft II window size (default is 1200).
        heuristic_ai: bool, optional
            Whether or not to use a non-learning heuristic AI (default False).
        heuristic_rest: bool, optional
            At any moment, restrict the actions of the heuristic AI to be
            chosen from actions available to RL agents (default is False).
            Ignored if heuristic_ai == False.
        debug: bool, optional
            Log messages about observations, state, actions and rewards for
            debugging purposes (default is False).
        """
        # Map arguments
        self.map_name = map_name
        map_params = get_map_params(self.map_name)
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        
        # custom code
        self.n_neutrals = map_params["n_neutrals"] 
        self.n_targets = self.n_enemies + self.n_neutrals

        self.episode_limit = map_params["limit"]
        self._move_amount = move_amount
        self._step_mul = step_mul
        self.difficulty = difficulty

        # init함수로 초기화한 정보를 입력한다. 지도정보와, 지도정보에 있는 맵파라미터 정보를 get_map_params로 불러와 map_params 에 입력한다
        # 다음다음다음다음 문단에서 해당정보를 각각의 맵과관련한 변수에 나누어서 저장하는 부분이 나온다
        # 여기서는 아군과 적군 에이전트 수와 에피소드의 타임리밋값, 그리고 타임스탭당 무브액션이 나왔을때 이동하는양(아까전에 하이퍼파라미터로 설정한값)
        # step_mul(전에 설명), 난이도 등등을 StarCraftEnv 클래스의 멤버변수에 각각 넣어준다

        # Observations and state
        self.obs_own_health = obs_own_health
        self.obs_all_health = obs_all_health
        self.obs_instead_of_state = obs_instead_of_state
        self.obs_last_action = obs_last_action
        self.obs_pathing_grid = obs_pathing_grid
        self.obs_terrain_height = obs_terrain_height
        self.obs_timestep_number = obs_timestep_number
        self.state_last_action = state_last_action
        self.state_timestep_number = state_timestep_number
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9

        self.obs_broadcast_info = obs_broadcast_info
        self.obs_communicate_info = obs_communicate_info
        # init 함수에서 초기화된 것들을 멤버변수에 넣어준다. 특이한 것은 n_obs_pathing과 n_obs_height 인데
        # n_obs_pathing은 에이전트가 이동하는 move_feature를 만들때 이동가능한지 아닌지를 판단하게 해주는 매뉴얼한 값이다(뒤에서 자세히 보기), 마찬가지로 n_obs_height 도 관련된 feature를 만들기 위한 매뉴얼한 값이다

        # Rewards args
        self.reward_sparse = reward_sparse
        self.reward_only_positive = reward_only_positive
        self.reward_negative_scale = reward_negative_scale
        self.reward_death_value = reward_death_value
        self.reward_win = reward_win
        self.reward_defeat = reward_defeat
        self.reward_scale = reward_scale
        self.reward_scale_rate = reward_scale_rate

        # Other
        self.game_version = game_version
        self.continuing_episode = continuing_episode
        self._seed = seed
        self.heuristic_ai = heuristic_ai
        self.heuristic_rest = heuristic_rest
        self.debug = debug
        self.window_size = (window_size_x, window_size_y)
        self.replay_dir = replay_dir
        self.replay_prefix = replay_prefix

        # Actions
        # self.n_actions_no_attack 을 6에서 7로 변경(어빌리티(스킬)사용용 차원)
        self.n_actions_no_attack = 7
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + self.n_enemies + self.n_neutrals

        # Map info
        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        ## unit_type_bits 는 smac_maps.py 에 정의된 값들인데, 해당 레지스트리에 에이전트 유닛병종이 1개이면 0,
        ## 2개이면 2,  3개이면 3으로 되어있다.(1은 왜 빠져잇는거지????)
        ## 여기에 대한 대답은 글로벌 스테이트에 정보를 줄때 self.unit_type_bits를 더해서 
        ## 각 에이전트가 가지는 정보량을 넣어주는데, 동종유닛간의 환경에서는 굳이 이를 1로 할필요가 없기 때문에 0으로 인코딩해서 넣어주고
        ## 2개 에이전트부터는 각각의 에이전트가 one-hot 으로 차원을 가져야하기 때문에 비트를 사용하는 것으로 보인다.
        self.map_type = map_params["map_type"]
        ## map_type 값도 smac_maps.py 에 정의되었는데, 이것은 특정 유닛들(거신, 히드라 등등)을 지도 타입을 저장한값이다.

        self.max_reward = (
            self.n_enemies * self.reward_death_value + self.reward_win
        )

        self.agents = {}
        self.enemies = {}
        # 뒤에 있는 init_units(self) 함수에 쓰인다. 이함수는 정해진 에이전트 숫자만큼 스타 클라이언트상에서 유닛을 생성하고 숫자가 맞게 생성되었는지 검토후에 안되었을시 에러메시지를 띄운다

        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        # 게임이 실행되면 해당 값을 에피소드와 타임스텝이 진행될때마다 증가시킨다
        # def step(self, actions): 에서 이를 1씩 증가시키는 연산자가 있다.
        # def reset(self): 에서 episode_steps 는 에피소드가 리셋됨에 따라 0으로 초기화된다

        self._obs = None
        # controller가 받아온 정보를 저장하는 객체이다
        self.battles_won = 0
        # 아래에 게임이 이기는지 졌는지를 부울값으로 확인하는 코드가있는데(해당코드는 api를 사용하는것으로 보임), 조건을 충족하면 이긴것으로하여 battles_won을 1씩 올린다
        self.battles_game = 0
        # 나중에 승률 계산하기 위해서 분모값으로 넣어주는 놈으로 보인다. 그런데 궁금한 것은 에피소드 카운터로 하지않는가? 중간에 초기화를 한다던지 이유가 잇겟지?(에피소드 카운터는 전역변수이기 때문에 별도로 선언해준것같음)
        self.timeouts = 0
        # 뒤에서 나오지만 에피소드가 리밋보다 길어지면 1 추가한다, 이게 비기는 게임에 대한 통계인듯하다
        self.force_restarts = 0
        # sc2 client를 강제 재실행할때 카운터 된다.(1씩 늘어남)

        self.last_stats = None
        # 검색해도 어디 쓰이는지 안나옴 ㅠㅠㅠ??????
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.killed_value_units = 0
        self.n_killed_units = 0
        self.total_damage_dealt = 0
        # def reward_battle(self): 함수에서 사용되며, 에이전트의 사망정보를 보상에 반영할때 쓰인다.
        self.delta_n_killed_unit = 0
        self.n_ally_alive = 0
        self.previous_n_ally_alive = 0

        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.previous_killed_value_units = None
        self.previous_damage_dealt = 0
        # def update_units(self): 에서 쓰이며, 타임스탭에 있는 액션이 실행되면 넥스트 스테이트에서 정보를 저장하기 위한 용도로 쓰임
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        # 에이전트의 마지막 행동을 저장하는 넘파이 텐서
        self._min_unit_type = 0
        # 뒤에보면 함수중에 def _init_ally_unit_types(self, min_unit_type): 이 있는데, 이 함수의 역할은 맵 타입마다 해당맵에 있는 유닛 id값이 다른데
        # 이를 컨트롤하기 위한 unit type 값을 각 맵별 각 유닛종류에 할당하는 역할을 한다. 이 함수에서 사용되는 변수이며, 마린맵같이 마린 1종류만 있는 맵에서는 마린에 초기화된 self._min_unit_type 값이 부여된다.


####
        self.marine_id = self.marauder_id = self.medivac_id = self.ghost_id = self.siegetank_id = self.siegetanksieged_id = self.scv_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0
        # 각 유닛 별로 id값과 관련된 변수를 선언하고 0으로 초기화한다
####
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0
        # 뒤에 launch 함수에서 사용됨
        self.terrain_height = None
        # 에이전트가 고도 정보를 받아와서 저장하는 변수
        self.pathing_grid = None
        # 에이전트가 이동가능 정보를 받아와서 저장하는 변수
        self._run_config = None
        # 스타크래프트 게임 버전 정보관련 변수
        self._sc2_proc = None
        # 뒤에보면 self._run_config.start(window_size=self.window_size, want_rgb=False) 와 같이 런치와 관련된 정보를 받아서 스타를 실행하는 함수이다
        self._controller = None
        # _controller라고 선언된 변수는 바로 밑에 launch 함수에서 observation에 게임정보를 던지는아이로 나온다.

        # Try to avoid leaking SC2 processes on shutdown
        atexit.register(lambda: self.close())
        # atexit 은 파이썬 표준라이브러리로 종료처리기라고 하며, 메모리 누수를 막기위한 것으로 보인다

    # 런치함수 스타2 실행. 게임을 돌아가게하는 가장상위에 있는 객체임
    # 기존에 선언 및 할당햇던 클래스 멤버 변수를 받아서 실행하는 설정값으로 활용함

        self.ally_state_attr_names = ['health', 'energy/cooldown', 'rel_x', 'rel_y',]
        self.enemy_state_attr_names = ['health', 'rel_x', 'rel_y']

        if self.shield_bits_ally > 0:
            self.ally_state_attr_names += ['shield']
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names += ['shield']

        if self.unit_type_bits > 0:
            bit_attr_names = ['type_{}'.format(bit) for bit in range(self.unit_type_bits)]
            self.ally_state_attr_names += bit_attr_names
            self.enemy_state_attr_names += bit_attr_names


    def _launch(self):
        """Launch the StarCraft II game."""
        self._run_config = run_configs.get(version=self.game_version)
        # run_configs.get 는 pysc2/run_configs/__init__.py에 있는 함수로 기본이 최신버전을 반환하는 함수이다
        _map = maps.get(self.map_name)
        # 맵변수를 선언하여 맵이름 받아옴, pysc2/maps/__init__.py 에 get=lib.get 으로 선언한 함수를 사용함
        # Setting up the interface
        interface_options = sc_pb.InterfaceOptions(raw=True, score=True)
        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=False)
        self._controller = self._sc2_proc.controller
        # _sc2_proc.controller 라는 애(pysc2 가 위에 선언한 멤버변수 _controller에 환경에서 얻은 모든 정보를 넘겨주는 역할을 한다.
        # run_config                controller가 게임전체의 정보 유닛들이 움직이는 정보 줌
        # Request to create the game   <------ 게임을 만들어라(종족 고르고, 여러 게임생성)
        create = sc_pb.RequestCreateGame(  # 유저가 된것
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path)),
            realtime=False,
            random_seed=self._seed)
        create.player_setup.add(type=sc_pb.Participant)  # 참여
        create.player_setup.add(type=sc_pb.Computer, race=races[self._bot_race],  # 종족
                                difficulty=difficulties[self.difficulty])
        self._controller.create_game(create)  # 게임을 만들어라

        join = sc_pb.RequestJoinGame(race=races[self._agent_race],
                                     options=interface_options)
        self._controller.join_game(join)

        game_info = self._controller.game_info()  # 게임 정보가 아마 다들어갈것
        map_info = game_info.start_raw  # 이게 무엇인지는 정확히 모르겟음
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:  # 이건 건들필요없어보임
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8))
            self.pathing_grid = np.transpose(np.array([
                [(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                for row in vals], dtype=np.bool))
        else:
            self.pathing_grid = np.invert(np.flip(np.transpose(np.array(
                list(map_info.pathing_grid.data), dtype=np.bool).reshape(
                    self.map_x, self.map_y)), axis=1))

        self.terrain_height = np.flip(  # 모든 맵영역에 높이 정보가 다 있음
            np.transpose(np.array(list(map_info.terrain_height.data))
                         .reshape(self.map_x, self.map_y)), 1) / 255


    def _calc_distance_mtx(self):
        # Calculate distances of all agents to all agents and enemies (for visibility calculations)
        dist_mtx = 1000 * np.ones((self.n_agents + self.n_targets, self.n_agents + self.n_targets))
        for i in range(self.n_agents + self.n_targets):
            for j in range(self.n_agents + self.n_targets):
                if j < i:    # 주대각선 아래는
                    continue # 전부 for문 이전에 초기화된 1000 값 유지
                elif j == i:                # 주대각선
                    dist_mtx[i, j] = 0.0    # 본인자신과의 거리는 0 으로 변경
                else:
                    if i >= self.n_agents:      # i 인덱스의 적군유닛을 가리키면 적군유닛 배열에서 해당하는 유닛값을 가져옴
                        unit_a = self.targets[i - self.n_agents]
                    else:
                        unit_a = self.agents[i] # i 인덱스가 아군을 가리키면
                    if j >= self.n_agents:
                        unit_b = self.targets[j - self.n_agents]
                    else:
                        unit_b = self.agents[j]
                    if unit_a.health > 0 and unit_b.health > 0:
                        dist = self.distance(unit_a.pos.x, unit_a.pos.y,
                                             unit_b.pos.x, unit_b.pos.y)
                        dist_mtx[i, j] = dist
                        if j < self.n_agents:
                            dist_mtx[j, i] = dist
        self.dist_mtx = dist_mtx

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()
            # 런처를 재실행하지 않고 게임내에서 남은 유닛 정리하고 2스텝가는 코드

        # Information kept for counting the reward
        
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        
        self.n_killed_units = 0
        self.killed_value_units = 0
        self.previous_killed_value_units = 0
        self.delta_n_killed_unit = 0



        # custom code
        self.previous_neutral_units = None
        
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))


        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()

            # 게임 에피소드의 첫단계에서 아군적군 개체에 대한 정보를 초기화 시키고 숫자가 맞는지 체크한 뒤에 
            # 이상없으면 1스텝 진행, 오류가 발생하면 런쳐를 재실행하거나 
            # 리셋(호출한 함수인 reset함수를 호출된 init_units 함수가 재귀적으로 다시호출)을 다시한번하는 코드
            self.init_units()

            self.set_visibility_matrix(self.obs_broadcast_info, self.obs_communicate_info)

        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug("Started Episode {}"
                          .format(self._episode_count).center(60, "*"))

        return self.get_obs(), self.get_state()


    # 런처를 재실행하지 않고 게임내에서 남은 유닛 정리하고 2스텝가는 코드
    def _restart(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            self._kill_all_units()
            self._controller.step(2)
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

    def full_restart(self):
        """Full restart. Closes the SC2 process and launches a new one. """
        self._sc2_proc.close()
        self._launch()

        self.previous_damage_dealt = 0
        self.total_damage_dealt = 0

        self.force_restarts += 1

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        actions_int = [int(a) for a in actions]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action)
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            return 0, True, {}

         # if self._episode_steps == 0 and self._episode_count == 0:
         #     print("targets length:::", len(self.targets))
         #     print(self.targets)

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()
         #self._calc_distance_mtx()
        self.set_visibility_matrix(self.obs_broadcast_info, self.obs_communicate_info)
         # if self._episode_steps == 6 and self._episode_count == 0:
         #     print("step:", self._episode_steps)
         #     print("targets length:::", len(self.targets))
         #     print(self.targets)


        #print("#epi", self._episode_count, "#step", self._episode_steps)

        #print(self.visibility_matrix)


        terminated = False
        reward = self.reward_battle()

        

        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1

        dead_enemies = self.n_killed_units
        info['dead_allies'] = dead_allies
        info['dead_enemies'] = dead_enemies

         #if self._episode_count % 10 == 0:
         # print("killunit", self.n_killed_units, "deltakill", self.delta_n_killed_unit, "de_en", dead_enemies, "count", self._episode_count, "step", self._episode_steps)

        if game_end_code is not None:
            # Battle is over
            terminated = True
         
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1
        # game_end_code 가 전달되면 terminated 변수에 True 값 저장, self.battles_game에 1을 올려서
        # 나중에 게임관련 통계를 낼때 사용함. self.win_counted 는 게임상에서의 승리가 아닌 판정으로 승리되면 증가하는 변수인듯함
        # 여튼 이기면 self.battles_won 에 1을 추가하고 self.reward_sparse 가 할당안됫으면 승리값을 reward 에 더해주고
        # sparse하면 1만 더해준다
        # 그리고 game_end_code 가 -1 이고 self.defeat_counted 가 false면 이를 True 로 바꿔준다
        # 만약 sparse 리워드 아니면 패배값을 리워드에 더해주고, sparse면 -1을 빼준다 


        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1
    
        # 에피소드 스텝이 에피소드 리밋보다 크면 에피소드를 종료상태로 만들고 게임관련 통계를 업데이트한다.
        # 그리고 타임아웃에 1을 추가한다. 뒤에서 보면 타임아웃이 1추가되면 비긴게임으로 치는것 같다
        # 아마도 제한시간내에 적을 다죽이거나 반대로 다죽지 않으면 비기는 것을 보는것같다 
    ## 디버그로 화면에 띄우기 위한 코드임. 실제로 잘동작하나 리플레이에서 재생이 안된다.
    #    str_reward = str(reward)
    #    for_reward = "{}".format(reward)
    #    print("reward is")
    #    print(reward)
    #    print(type(reward))
    #    print("str_reward is")
    #    print(str_reward)
    #    print(type(str_reward))
    #    print("for_reward is")
    #    print(for_reward)
    #    print(type(for_reward))
    ##    vp_pos = sc_common.Point(x=16,y=16,z=20)
    #    print(vp_pos)
    ##    wp_pos = sc_common.Point(x=14,y=14,z=12)
    #    print(wp_pos)
    #    color = d_pb.Color(r=0,g=0,b=0)
    #    text = d_pb.string(text=for_reward)
    #    debug_command = [  d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))   ]
    #           cmd = d_pb.DebugCommand(
    #               create_unit=d_pb.DebugCreateUnit(
    #                   unit_type=unit_type_id,
    #                   owner=1,
    #                   pos=sc_pos,
    #                   quantity=num))
    #           cmds.append(cmd)
    #           cmd = d_pb.DebugCommand(
    #               create_unit=d_pb.DebugCreateUnit(
    #                   unit_type=unit_type_id,
    #                   owner=2,
    #                    pos=sc_pos,
    #                    quantity=num))
    #            cmds.append(cmd)
    #           self.n_enemies += num
    #        self._controller.debug(cmds)
    #            cmd = d_pb.DebugCommand(
    #        game_state=d_pb.DebugGameState.control_enemy)
    #    step_success = self.try_controller_step(fn=lambda: self._controller.debug([cmd]),   n_steps=4)

    #    debug_command = d_pb.DebugDraw(text=d_pb.DebugText(color=white, text=str_reward))
    #    debug_command = d_pb.DebugText(Color='red', text="str_reward",virtual_pos=11, world_pos=12, size=13)
    #    debug_command = d_pb.DebugText(color=color, text=str_reward, virtual_pos=vp_pos, world_pos=wp_pos, size=10)
    #    debug_command = d_pb.DebugDraw(text= d_pb.DebugText(text=for_reward))
    # text = d_pb.DebugText("for_reward"), virtual_pos = vp_pos, world_pos = wp_pos )

    #    command = d_pb.DebugCommand(
    #        score = d_pb.DebugSetScore(score = 2.0))
    #    print(command)


    #    spheres = []
    #    sphere = d_pb.DebugSphere(p=wp_pos, r = 3.0)
    #    spheres = spheres.append(sphere)
    #    draw = d_pb.DebugDraw(spheres = spheres)
        
    ##    color = d_pb.Color(r = 255, g = 255, b = 0)
    ##    texts = []
    #    text = d_pb.DebugText(text = "draw text is difficult", virtual_pos = vp_pos, world_pos=wp_pos, color = color, size = 40)
    ##    text = d_pb.DebugText(text = "draw text is difficult", world_pos = wp_pos, color = color, size = 40)
    ##    texts.append(text)
    ##    print("text~~~~~")
    ##    print(text)
    ##    print("-----------------")
    ##    print("texts~~~~~")
    ##    print(texts)
    ##    print("-----------------")
    ##    draw = d_pb.DebugDraw(text = texts)
    ##    print("draws~~~~~")
    ##    print(draw)
    ##    print("-----------------")
    ##    cmds = []
    ##    cmd = d_pb.DebugCommand(draw = draw)
    ##    print(cmd)
    ##    cmds.append(cmd)
    ##    self._controller.debug(cmds) , channel=sc_pb.ActionChat.Broadcast, 




        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, '-'))
        # self.debug 라는 변수가 참값이면 리워드 값에 대해서 로깅하도록 한다
        # 요부분은 pymarl에서 어떻게 쓰이는지, 그리고 활용가능한건지 알아보고싶다

        if terminated:
            self._episode_count += 1
        # 에피소드가 끝났을때 에피소드를 1씩 카운트한다(통계정보 생성에 활용)

        if self.reward_scale:
            reward *= self.reward_scale_rate / self.max_reward
        # reward = (reward * self.reward_scale_rate) / self.max_reward 를 한것으로
        # self.max_reward 는 쉴드로 채워지는 부분을 제외하고 획득가능한 총보상합이다
        # 받은 보상을 스케일과 곱해서 맥스리워드로 나눈것은 스케일을 줄이는 것인데, reward_scale이 참으로 설정되면 실행된다 



         # if self._episode_count % 100 == 0 and self._episode_steps == 50 :
         #     message = "Qval#1 is "
         #     message = message + str(self._episode_count)
         #     action_chat = sc_pb.ActionChat(message = message)
         #     action = sc_pb.Action(action_chat=action_chat)
         #     actions = []
         #     actions.append(action)
         #     req_actions = sc_pb.RequestAction(actions=actions)
         #     self._controller.actions(req_actions)


        return reward, terminated, info
        # 스텝함수가 실행되고 종료되면 1-step이 지난것이며 에피소드 종료후에 관련된 변수들을 업데이트하고 리턴한다



    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert avail_actions[action] == 1, \
            "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        # 에이전트로부터 행동을 받아서(아마도 뉴럴넷이 출력하는 거겟지,) 게임내에서 


        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))
        # action 이 0 이면 죽은 에이전트에 한하여 아무 움직임도 취하지 않는다
        # action 이 1 이면 스톱을 하는데, 관련 api로 부터 불러와서 cmd 변수에 이를 저장한다
        # 그리고 ability_id 에 57번 line 즈음에 action set의 이름을 키로, 실제 id에 해당하는 것을
        # 밸류형태로 저장했는데 스톱에 해당하는 키로 해당 밸류값을 얻어서 ability_id 에 이를 할당한다
        # 또한 유닛태그를 받아서 넣어준다
        # 실시간으로 제어하기때문에 queue_command 는 아래나오는 액션들도 전부 False 값으로 되어있다.
        # r_pb.ActionRawUnitCommand 라는 애가 starcraft 클라이언트에 튜플(??) 형태로 자료를 전달하여 게임을 구동하는 것으로 보인다

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))
        # action 이 2 이면 북쪽(y좌표에 + 해주는 방식)으로 self._move_amount(__init__에서 2로 설정) 만큼 이동한다.
        



        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))
        # action 이 3 이면 남쪽(y좌표에 - 해주는 방식)으로 self._move_amount(__init__에서 2로 설정) 만큼 이동한다.



        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))
        # action 이 4 이면 북쪽(x좌표에 + 해주는 방식)으로 self._move_amount(__init__에서 2로 설정) 만큼 이동한다.



        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False)
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))
        # action 이 5 이면 서쪽(ㅌ좌표에 - 해주는 방식)으로 self._move_amount(__init__에서 2로 설정) 만큼 이동한다.

        elif action == 6:
            # use ability eg siege mode(siege tank) or steady targeting(ghost)
            if self.map_type == "plain_neut_tank" and  unit.unit_type == 919:
                cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["morph_siegemode"],
                unit_tags=[tag],
                queue_command=False)
            elif self.map_type == "plain_neut_tank" and  unit.unit_type == 918:
                cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["morph_unsiege"],
                unit_tags=[tag],
                queue_command=False)
#            # 맨위에 선언한 "actions" dictionary에서 MORPH_SIEGEMODE에 해당하는 인코딩된 정수(시즈 388,언시즈 390)을 꺼내서 ability_id에 해당값 할당
#            # ability_id는 매개변수 역할을 하여 전달하는듯
            else:
                cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False)

            if self.debug:
                logging.debug("Agent {} {} ".format(
                    a_id, action_name))
            # 다른 이동관련 액션처럼 디버그를 넣어준다


        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                target_unit = self.agents[target_id]
                action_name = "heal"
            else:
                target_unit = self.targets[target_id]
                action_name = "attack"

        # 나머지 action은 맵에 따라, 그리고 유닛타입에 따라 메디박이면 아군에이전트 id를 받아 heal을 하고,
        # 아니면 적군 id를 받아 공격을 하는 액션을 한다 

            action_id = actions[action_name]
            target_tag = target_unit.tag
            # action_id 라는 리스트에 액션 이름을 넣어주고 타겟의 테그를 넣어준다.

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)
            # 바로 위에서 저장한 액션 id와, 타겟 tag를 star2 api 로 전달하는 cmd변수에 저장한다.
            if self.debug:
                logging.debug("Agent {} {}s unit # {}".format(
                    a_id, action_name, target_id))
            # 다른 이동관련 액션처럼 디버그를 넣어준다

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

        # 아까 저장했던 cmd 를 관련 api의 매개변수에 넣어주고 리턴한다




    def get_agent_action_heuristic(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        # unit 이란 변수가 선언과 함꼐 저장된다
        tag = unit.tag

        target = self.heuristic_targets[a_id]
        if unit.unit_type == self.medivac_id:
            if (target is None or self.agents[target].health == 0 or
                    self.agents[target].health == self.agents[target].health_max):
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for al_id, al_unit in self.agents.items():
                    if al_unit.unit_type == self.medivac_id:
                        continue
                    if (al_unit.health != 0 and
                            al_unit.health != al_unit.health_max):
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             al_unit.pos.x, al_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = al_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['heal']
            target_tag = self.agents[self.heuristic_targets[a_id]].tag
        else:
            if target is None or self.enemies[target].health == 0:
                min_dist = math.hypot(self.max_distance_x, self.max_distance_y)
                min_id = -1
                for e_id, e_unit in self.enemies.items():
                    if (unit.unit_type == self.marauder_id and
                            e_unit.unit_type == self.medivac_id):
                        continue
                    if e_unit.health > 0:
                        dist = self.distance(unit.pos.x, unit.pos.y,
                                             e_unit.pos.x, e_unit.pos.y)
                        if dist < min_dist:
                            min_dist = dist
                            min_id = e_id
                self.heuristic_targets[a_id] = min_id
                if min_id == -1:
                    self.heuristic_targets[a_id] = None
                    return None, 0
            action_id = actions['attack']
            target_tag = self.enemies[self.heuristic_targets[a_id]].tag

        action_num = self.heuristic_targets[a_id] + self.n_actions_no_attack

        # Check if the action is available
        if (self.heuristic_rest and
                self.get_avail_agent_actions(a_id)[action_num] == 0):

        ## 휴리스틱_레스트는 위의 불리언 값으로 되어있으며, 위에 설명은 RL에이전트의 행동으로부터 제약을 가할수 있는지 여부인데
        ## 기본적으로는 false값으로 되어있다. 제약이 없어야(0값이어야) 아래 if문 밑의 내용이 수행되는 것같다
        ## get_avail_agent_actions*() 함수는 이모듈 뒷부분에 정의되어있는데, 동서남북으로 이동하는 것이 가능한지 체크하는 역할을 하는 함수이다
        ## 또한 이함수는 사정거리와 시야거리 사이에 있는 적에 대한 공격 여부를 판단하기도 한다.
        ## 그래서 만약에 공격이 불가능하다면 아래 코드가 실행되는데 공격하거나 치료하는게 아니라 상대방을 향해 이동하도록한다
        ## 상대방과의 x, y 좌표 값의 차이의 절댓값을 바탕으로 더큰 discrepancy 를 줄이기 위한 방향으로 이동하도록 코드가 짜여져있고
        ## 위의 액션관련함수처럼 cmd변수에 저장한다.

            # Move towards the target rather than attacking/healing
            if unit.unit_type == self.medivac_id:
                target_unit = self.agents[self.heuristic_targets[a_id]]
            else:
                target_unit = self.enemies[self.heuristic_targets[a_id]]

            delta_x = target_unit.pos.x - unit.pos.x
            delta_y = target_unit.pos.y - unit.pos.y

            if abs(delta_x) > abs(delta_y):  # east or west
                if delta_x > 0:  # east
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x + self._move_amount, y=unit.pos.y)
                    action_num = 4
                else:  # west
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x - self._move_amount, y=unit.pos.y)
                    action_num = 5
            else:  # north or south
                if delta_y > 0:  # north
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y + self._move_amount)
                    action_num = 2
                else:  # south
                    target_pos = sc_common.Point2D(
                        x=unit.pos.x, y=unit.pos.y - self._move_amount)
                    action_num = 3

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions['move'],
                target_world_space_pos=target_pos,
                unit_tags=[tag],
                queue_command=False)
        else:
            # Attack/heal the target
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False)
        ## 위의 if문의 else로 되어있는 부분으로 공격이 가능한 거리이면 공격을 취하는 것이다.

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action, action_num

        # 유닛간의 거리를 계산해서 가장 가까운 적부터 공격하는 아군 휴리스틱 AI이다. 이는 각종 알고리즘의 베이스라인을 설정하기 위해 쓰이며
        # 가용한 액션으로부터 실제 액션을 샘플링하는 코드는 pymarl에 여러군데있는데 기본적으로 q-value를 출력하는 rnn_agent.py와
        # basic_controller.py 에 액션 샘플링하는 코드가 정의 되어있으며, 이 함수를 콜해서 실행하는 코드는 episode_runner.py 에 있다.


    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        # Custom
        delta_death_value = 0
        delta_enemy_health_value = 0




        # 보상관련 함수로써 클래스의 멤버변수인 self.reward_sparse 가 True 이냐 False이냐에 따라 reward 함수가 달라진다
        # True 이면 보상이 sparse하기 때문에 위에서 작성된 코드대로 이겻을때 1/-1 의 리워드만 가능하고 
        # 실질적으로 이 함수 자체는 0값을 반환한다. 결국 나머지 게임내 변수들은 reward값에 영향을 주지 않는다
        # 반면 False이면 reward, delta_deaths, delta_ally, delta_enemy 는 사망한 아[적]군 유닛 수인데,
        # 이것들이 보상에 영향을 미치도록 위에서 0으로 변수선언과 함께 초기화 하고 있다.



        neg_scale = self.reward_negative_scale
        # 보상이 음수값을 갖는것을 설정했을때 리워드의 스케일을 조정하는 변수이다


        # update deaths
        for al_id, al_unit in self.agents.items():
            if not self.death_tracker_ally[al_id]:
                # did not die so far
                prev_health = (
                    self.previous_ally_units[al_id].health
                    + self.previous_ally_units[al_id].shield
                )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )
        





        delta_death_value += self.reward_death_value * self.delta_n_killed_unit     
        self.previous_damage_dealt = deepcopy(self.total_damage_dealt)
        self.total_damage_dealt = self._obs.observation.score.score_details.total_damage_dealt.life
        delta_enemy_health_value = self.total_damage_dealt - self.previous_damage_dealt



         # for e_id, e_unit in self.enemies.items():
         #     # if문 이하는 e_id 인덱스에 해당하는 적군유닛이 살아있는 경우 실행됨  
         #     if not self.death_tracker_enemy[e_id]:
         #         prev_health = (
         #             self.previous_enemy_units[e_id].health
         #             + self.previous_enemy_units[e_id].shield
         #         )
         #         # 전state에서 살아있었던 유닛의 경우 health가 0이되면 
         #         # death_tracker_enemy의 값을 1로 바꿔주어 죽은 유닛으로 바꿔준다.
         #         # 여기에 조건을 추가하여 업데이트 된것만 죽은것으로 처리하도록 한다.
         #         if e_unit.health == 0:
         #             self.death_tracker_enemy[e_id] = 1
         #             delta_deaths += self.reward_death_value
         #             delta_enemy += prev_health
         #         else:
         #             delta_enemy += prev_health - e_unit.health - e_unit.shield

        if self.reward_only_positive:
            reward = abs(delta_enemy_health_value + delta_death_value)  # shield regeneration
        else:
            reward = delta_enemy_health_value + delta_death_value - delta_ally




        return reward



    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions
    
    ## 전체 엑션갯수를 받아오는 함수이다. 환경에서 생성한 정보를 바탕으로 에이전트의 액션 디멘션을 맞추기 위해서 self.n_actions 를 반환하는 것 같다
    ## 세부 구현은 pymarl에 있는 것으로 보인다(검색하면 나옴)


    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)
    # python 등 언어에 공통적으로 있는 빗변계산 함수로, 여기서는 두 점사이의 유클리디안 거리를 계산하는데에 쓰인다
    # 뒤에서 자주 등장한다

    
    # 살짝 수정함(매개변수를 agent_id에서 unit으로 변경)
    def unit_shoot_range(self, unit):
        """Returns the shooting range for an agent."""
        unit = unit
        if unit.unit_type == 3225:   # marine
            return 6
        elif unit.unit_type == 3224:   # marauder
            return 7
        elif unit.unit_type == 919:   # siegetank_tankmode
            return 8
        elif unit.unit_type == 918:   # siegetank_siegemode
            return 17
        else:
            return 6
    # 실제 마린 사거리는 5이고 다른 유닛들도 6과는 다른 경우가 많은데, 어떤 이유로 사거리를 이와같이 설정했는지는 잘 모르겠다
    # 다만 가능한 액션셋을 뽑는 함수에서 이 함수를 호출하여 6안에 있으면 공격가능한 것으로 판단한다.
    # 혹시 실제 사정거리가 이보다 짧은 에이전트가 6과 실제 사정거리 사이에서 어떤 행동을 보이는지 알필요가 있어보인다

    def unit_sight_range(self, agent_id):
        """Returns the sight range for an agent."""
        return 9
    # 시야거리를 반환하는 함수이다, SMAC은 비전을 켜서(전장의 안개 제거) 맵전체를 다보이게 한다음에 각 유닛들에게는 partial 한 관측정보를
    # 이 함수의 범위만큼 제한해서 제공하는 것 같다

    def unit_max_cooldown(self, unit):
        """Returns the maximal cooldown for a unit."""
        switcher = {
            self.marine_id: 15,
            self.marauder_id: 25,
            self.medivac_id: 200,  # max energy
            self.stalker_id: 35,
            self.zealot_id: 22,
            self.colossus_id: 24,
            self.hydralisk_id: 10,
            self.zergling_id: 11,
            self.baneling_id: 1,
            self.ghost_id: 25,
            self.siegetank_id: 20,
            self.siegetanksieged_id: 70,
            self.scv_id: 25
        }
        return switcher.get(unit.unit_type, 15)
        ## 의료선을 제외하고는 게임내에서 공격 연사속도를 의미하며, 
        ## 연사값단위는 1게임 프레임 단위로 결정된다. 15프레임을 게임속도 보통으로 할때는 마린이 1초에 한대씩 때린다
        ## 24프레임이 게임속도 매우빠름인데, 이때는 0.625이다(빠름은 18프레임으로 0.8333)
        ## return switcher.get(x,y) 는 x 키에 해당하는 value를 반환하고 만약 x 키에 해당하는 정보가 없으면 y를 반환한다는 의미다
        ## 고로 위에서는 self~~에 해당하는 키와 일치하는 value(오른쪽값)를 리턴한다

    def save_replay(self):
        """Save a replay."""
        prefix = self.replay_prefix or self.map_name
        replay_dir = self.replay_dir or ""
        replay_path = self._run_config.save_replay(
            self._controller.save_replay(), replay_dir=replay_dir, prefix=prefix)
        logging.info("Replay saved at: %s" % replay_path)
        # 리플레이를 저장하는 함수로 관련된 멤버변수를 통해 self._run_config.save_replay()에 관련 값을 전달한다.


    def unit_max_shield(self, unit):
        """Returns maximal shield for a given unit."""
        if unit.unit_type == 74 or unit.unit_type == self.stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == self.zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == self.colossus_id:
            return 150  # Protoss's Colossus
        # 유닛 타입별로 최대 쉴드량을 정하는 코드이다


    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False
        ## 이동 가능한지 아닌지를 판단하는 코드로 클래스 멤버변수인 _move_amount 를 기준으로하여 4방향이 이동가능한지를 방향별로 판단한다


    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points
        ## _move_amount 와 _move_amount 의 2배수 만큼 동서남북 방향으로 주변지점 좌표값을 저장한다

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return (0 <= x < self.map_x and 0 <= y < self.map_y)
        # 주어진 x, y 좌표가 맵의 경계선 안에 있는 지를 체크하는 함수이다. can_move()함수와 get_surrounding_pathing, get_surrounding_height 함수 등에서 사용된다.


    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [
            self.pathing_grid[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals
        # self.pathing_grid (launch 함수가 실행될때 맵정보로부터 이동가능불가여부를 전달받아 저장하는 변수)의 지도의 전역데이터에서
        # 바로 위에위에 선언된 함수인 get_surrounding_points()가 리턴하는 값(에이전트의 동서남북 방향 self._move_amount의 2배수)의 범위만큼
        # 필터링한 정보를 얻어온다. (agent의 지형의 이동가능불가 관측값은 로컬 정보이므로 범위를 제한하는 듯하다)
        # 다만 기준을 시야범위로 한것이 아니라 무브 어마운트로 한것은 의문점이다.
        # 바로 뒤에뒤에 get_obs_agent() 함수 (실질적으로 관측을 담당하는 함수, 관측과 관련된 모든 함수들을 실행시켜주는)에서 사용되어 에이전트가 이동가능 불가여부를 체크한다
        # 뒤에 멤버변수 값에 불리언 형태로 저장된 obs_pathing_grid 를 if문에 따라 참이면 agent에 이동가능불가 정보를 넣어주고
        # False면 넣지 않는데 쓰인다.

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [
            self.terrain_height[x, y] if self.check_bounds(x, y) else 1
            for x, y in points
        ]
        return vals
        # 바로 위의 서라운딩 패씽과 거의 유사한 방식이며 얻어오는 정보가 높이정보이고 불리언이아닌 실수값이라는 차이점만 있다.


    ## 에이전트 id를 입력으로 받아 해당 에이전트의 관측을 넘겨주는 함수이다.
    # 이동가능여부와, 높이 및 패씽 그리드 정보를 넘겨준다
    # 적군의 피쳐인 공격가능여부, 체력과 상대적 x,y좌표는 어떤지, 쉴드, 유닛 타입정보를 환경에서 받는다
    # 아군의 피쳐인 보이는지여부, 거리, 상대적 x,y좌표, 쉴드, 유닛 타입정보 등을 환경에서 받는다
    # 에이전트 자신의 피쳐인 체력, 쉴드, 유닛타입을 받는다
    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        unit = self.get_unit_by_id(agent_id)

        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        neutral_feats_dim = self.get_obs_neutral_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        visibility_mask = 0

        # custom code
        neutral_feats = np.zeros(neutral_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        # 에이전트의 피쳐차원을 계산한 뒤에 이를 넘파이 배열의 인자로 넣어서 피쳐벡터로 만들어준다
        # print("neut feat", np.shape(neutral_feats),"===")
        # print("enemy feat", np.shape(enemy_feats),"===")



        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            if self.obs_terrain_height:
                z = unit.pos.z
            sight_range = self.unit_sight_range(agent_id)

            # 1. Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            #### 요기 get_avail_agent_actions 에서 문제발생
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[
                    ind: ind + self.n_obs_pathing
                ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)
             # 유닛체력(zero or positive)을 기준으로 positive면 if문이 실행됨
             # 이동가능한 것을 계산하여 move_feats에다가 해당 정보를 넣어준다 


            # 2. Enemy features Matrix
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                if self.obs_terrain_height:
                    e_z = e_unit.pos.z
                    dist = self.distance(x, y, e_x, e_y)
                    visibility_mask = self.visibility_matrix[agent_id][self.n_agents + e_id]
                    if (
                        visibility_mask == 1 and e_unit.health > 0
                    ):  # visible and alive
                        # Sight range > shoot range
                        enemy_feats[e_id, 0] = avail_actions[
                            self.n_actions_no_attack + e_id
                        ]  # available
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (
                            e_x - x
                        ) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (
                            e_y - y
                        ) / sight_range  # relative Y
                        enemy_feats[e_id, 4] = (e_z - z) / 2   # relative Z
    
                        ind = 5
                        if self.obs_all_health:
                            enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                            )  # health
                            ind += 1
                            if self.shield_bits_enemy > 0:
                                max_shield = self.unit_max_shield(e_unit)
                                enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                                )  # shield
                                ind += 1
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(e_unit, False)
                            enemy_feats[e_id, ind + type_id] = 1  # unit type

                else:
                    dist = self.distance(x, y, e_x, e_y)
                    visibility_mask = self.visibility_matrix[agent_id][self.n_agents + e_id]
                    if (
                        visibility_mask == 1 and e_unit.health > 0
                    ):  # visible and alive
                        # Sight range > shoot range
                        enemy_feats[e_id, 0] = avail_actions[
                            self.n_actions_no_attack + e_id
                        ]  # available
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (
                            e_x - x
                        ) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (
                            e_y - y
                        ) / sight_range  # relative Y
                        ind = 4
                        if self.obs_all_health:
                            enemy_feats[e_id, ind] = (
                                e_unit.health / e_unit.health_max
                            )  # health
                            ind += 1
                            if self.shield_bits_enemy > 0:
                                max_shield = self.unit_max_shield(e_unit)
                                enemy_feats[e_id, ind] = (
                                    e_unit.shield / max_shield
                                )  # shield
                                ind += 1
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(e_unit, False)
                            enemy_feats[e_id, ind + type_id] = 1  # unit type
                # 적군 피쳐를 계산하는 부분으로 적이 우선 시야에 있는지 없는지 여부, for문이 모든 적에이전트 식별정보에 대해 돌아간다 
                # 시야에 있을때는 피쳐값을 계산하여 넘겨주며, 이때는 공격가능여부를 또한 판단하여 해당 정보를 넘겨준다
                # 체력과 쉴드 유닛타입정보 등을 넘겨준다


            # 3. Ally features Matrix(에이전트가 아군을 관측한 피쳐)
            al_ids = [
                al_id for al_id in range(self.n_agents) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                if self.obs_terrain_height:
                    al_z = al_unit.pos.z
                    dist = self.distance(x, y, al_x, al_y)
                    visibility_mask = self.visibility_matrix[agent_id][al_id]
                    if (
                        visibility_mask and al_unit.health > 0
                    ):  # visible and alive
                        ally_feats[i, 0] = 1  # visible
                        ally_feats[i, 1] = dist / sight_range  # distance
                        ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y
                        ally_feats[i, 4] = (al_z - z) / 2   # relative Z
    
                        ind = 5
                        if self.obs_all_health:
                            ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                            )  # health
                            ind += 1
                            if self.shield_bits_ally > 0:
                                max_shield = self.unit_max_shield(al_unit)
                                ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                                )  # shield
                                ind += 1
                            
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(al_unit, True)
                            ally_feats[i, ind + type_id] = 1
                            ind += self.unit_type_bits
    
                        if self.obs_last_action:
                            ally_feats[i, ind:] = self.last_action[al_id]
                else:
                    dist = self.distance(x, y, al_x, al_y)
                    visibility_mask = self.visibility_matrix[agent_id][al_id]
                    if (
                        visibility_mask and al_unit.health > 0
                    ):  # visible and alive
                        ally_feats[i, 0] = 1  # visible
                        ally_feats[i, 1] = dist / sight_range  # distance
                        ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y
                        ind = 4
                        if self.obs_all_health:
                            ally_feats[i, ind] = (
                                al_unit.health / al_unit.health_max
                            )  # health
                            ind += 1
                            if self.shield_bits_ally > 0:
                                max_shield = self.unit_max_shield(al_unit)
                                ally_feats[i, ind] = (
                                    al_unit.shield / max_shield
                                )  # shield
                                ind += 1                      
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(al_unit, True)
                            ally_feats[i, ind + type_id] = 1
                            ind += self.unit_type_bits
                        if self.obs_last_action:
                            ally_feats[i, ind:] = self.last_action[al_id]

                # 아군 피쳐정보도 적군과 유사하게 시야에서 보이는지 여부와 살아있는지 여부를 체크하고
                # 거리와 상대적 x,y 좌표를 계산하고 체력 쉴드를 계산하고, 유닛타입을 확인하고, 마지막행동이 참이면 해당정보도 받는다. 그런데 기본값을 거짓으로 되어있다.
                # 체력관련 값들은 현재체력값에서 최대체력 값을 나눈것으로 들어간다. 이것도 어떤 이유가 있겠지??
                # 쉴드는 self.shield_bits_ally 가 양수이면(1이면) 프로토스이고 그러면 체력과 같이 맥스쉴드로 현재쉴드를 나눈 값을 계산하여 넣어준다
                # 그러면 상대 유닛이 프로토스일때랑 테란유닛일때랑 피쳐정보의 차원이 다른가? 이건 확인해볼것!




            # 4. Neutral features Matrix (Custom code)
            for neut_id, neut_unit in self.neutrals.items():
                neut_x = neut_unit.pos.x
                neut_y = neut_unit.pos.y
                if self.obs_terrain_height:
                    neut_z = neut_unit.pos.z
                    dist = self.distance(x, y, neut_x, neut_y)
                    visibility_mask = self.visibility_matrix[agent_id][self.n_agents + self.n_enemies + neut_id]
    
                    if (
                        visibility_mask and neut_unit.health > 0
                    ):  # visible and alive
                        neutral_feats[neut_id, 0] = avail_actions[
                            self.n_actions_no_attack + self.n_enemies + neut_id
                        ]  # available
                        neutral_feats[neut_id, 1] = dist / sight_range  # distance
                        neutral_feats[neut_id, 2] = (
                            neut_x - x
                        ) / sight_range  # relative X
                        neutral_feats[neut_id, 3] = (
                            neut_y - y
                        ) / sight_range  # relative Y
                        neutral_feats[neut_id, 4] = (
                            neut_z - z
                        ) / 2      # relative Z
    
                        ind = 5
                        if self.obs_all_health:
                            neutral_feats[neut_id, ind] = (
                                neut_unit.health / neut_unit.health_max
                            )  # health
                            ind += 1
                         #    ind = 6
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(neut_unit, False)
                            neutral_feats[neut_id, ind + type_id] = 1  # unit type
                else:
                    dist = self.distance(x, y, neut_x, neut_y)
                    visibility_mask = self.visibility_matrix[agent_id][self.n_agents + self.n_enemies + neut_id]
    
                    if (
                        visibility_mask and neut_unit.health > 0
                    ):  # visible and alive
                        neutral_feats[neut_id, 0] = avail_actions[
                            self.n_actions_no_attack + self.n_enemies + neut_id
                        ]  # available
                        neutral_feats[neut_id, 1] = dist / sight_range  # distance
                        neutral_feats[neut_id, 2] = (
                            neut_x - x
                        ) / sight_range  # relative X
                        neutral_feats[neut_id, 3] = (
                            neut_y - y
                        ) / sight_range  # relative Y
    
                        ind = 4
                        if self.obs_all_health:
                            neutral_feats[neut_id, ind] = (
                                neut_unit.health / neut_unit.health_max
                            )  # health
                            ind += 1
                         #    ind = 5
                        if self.unit_type_bits > 0:
                            type_id = self.get_unit_type_id(neut_unit, False)
                            neutral_feats[neut_id, ind + type_id] = 1  # unit type




            # 5. Own features (에이전트 본인을 스스로 관측한 것에 대한 피쳐), 에이전트의 절대좌표값을 피쳐에 넣어주었음
            own_feats[0] = unit.pos.x
            own_feats[1] = unit.pos.y
            if self.obs_terrain_height:
                own_feats[2] = unit.pos.z
                ind = 3
                if self.obs_own_health:
                    own_feats[ind] = unit.health / unit.health_max
                    ind += 1
                    if self.shield_bits_ally > 0:
                        max_shield = self.unit_max_shield(unit)
                        own_feats[ind] = unit.shield / max_shield
                        ind += 1
                # self.unit_type_bits 는 smac_maps.py에서 받아온 unit_type_bits 를 그대로 멤버변수가 넘겨 받은것임(실행되는 맵의 정보에 따라 바뀜)
                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(unit, True)
                    own_feats[ind + type_id] = 1

            else:
                ind = 2
                if self.obs_own_health:
                    own_feats[ind] = unit.health / unit.health_max
                    ind += 1
                    if self.shield_bits_ally > 0:
                        max_shield = self.unit_max_shield(unit)
                        own_feats[ind] = unit.shield / max_shield
                        ind += 1
                # self.unit_type_bits 는 smac_maps.py에서 받아온 unit_type_bits 를 그대로 멤버변수가 넘겨 받은것임(실행되는 맵의 정보에 따라 바뀜)
                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(unit, True)
                    own_feats[ind + type_id] = 1
            # 자기자신에 대한 피쳐로 체력과 쉴드 그리고 유닛타입을 저장한다


        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),

                # custom code
                neutral_feats.flatten(),
                
                own_feats.flatten(),
            )
        )
        # 관측으로 획득한 개별적인 feature 들을 concat 해주어서 하나의 agent_obs 텐서로 만드는 과정이다

        if self.obs_timestep_number:
            agent_obs = np.append(agent_obs,
                                  self._episode_steps / self.episode_limit)
        ## self.obs_timestep_number은 init 함수에서 초기화되는데 기본값은 false 이므로 일반적으로 관측정보에 타임스텝정보를 넣어주지는않는것 같다.

        if self.debug:
            logging.debug("Obs Agent: {}".format(agent_id).center(60, "-"))
            logging.debug("Avail. actions {}".format(
                self.get_avail_agent_actions(agent_id)))
            logging.debug("Move feats {}".format(move_feats))
            logging.debug("Enemy feats {}".format(enemy_feats))
            logging.debug("Ally feats {}".format(ally_feats))
            logging.debug("Own feats {}".format(own_feats))

        return agent_obs
        ## starcraft2.py 전반에 logging.debug가 사용되어있는데,
        ## 여기있는 starcraft2.py의 멤버변수인 self.debug 가 pymarl 에서 sc2.yaml 의 debug인자 값을 true 로 하면 마찬가지로 true 값이되어
        ## if문이 실행되는데 absl에서 임포트된 함수가 화면에 입력으로 들어온 정보들을 모델훈련시 터미널에 출력시켜준다


    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs


    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
                np.float32
            )
            return obs_concat
            # self.obs_instead_of_state 는 처음 init에서 기본값이 false 로 설정되는데 
            # 개별 에이전트의 local한 observation 정보를 global state information 으로 쓸지 여부를 결정하는변수이다
            # True로 설정되면 위의 if문 이하가 실행되는데 observation을 concat하여 리턴한다
        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits
        ## 아래에 글로벌 스테이트를 만들때 ally와 enemy 각각의 차원은 
        ## number of feature 인듯 아[적]군 에이전트 숫자 * 개별에이전트가 가지는 정보차원인데, 이를 계산하기 위한 식으로 볼 수 있다.

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))
        ## 넘파이 배열로 글로벌 스테이트 텐서를 생성하며, 위에 적은 크기로 스테이트 사이즈를 만들고 0으로 초기화한다


        center_x = self.map_x / 2
        center_y = self.map_y / 2
        ## 맵의 중심부 좌표(절대적인 값)를 생성한다. 맵사이즈를 2로 나눈값이 맵 중앙의 좌푯값이다


        # 반복문이 돌면서 아군 유닛에 대해 순차적으로 실행한다
        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit)    
                # 체력이 있는 아군 유닛에 대해 좌표값을 저장하고
                # 최대 쿨타임을 저장한다 

                ally_state[al_id, 0] = (
                    al_unit.health / al_unit.health_max
                )  # health
                # 글로벌 스테이트 정보의 특정에이전트(for문에 의해 인덱스가 증가하는)의 텐서 두번째 차원(1번째)의
                # 벡터의 첫번째(0번째) 차원에 현재 체력/맥스 체력 값을 저장한다
                if (
                    self.map_type == "MMM"
                    and al_unit.unit_type == self.medivac_id
                ):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (
                        al_unit.weapon_cooldown / max_cd
                    )  # cooldown
                #맵에 따라 "MMM"맵이고 의료선이면 유닛의 마나(에너지)/맥스마나(에너지 값을) 를
                # 글로벌 스테이트 정보의 특정에이전트(for문에 의해 인덱스가 증가하는)의 텐서 두번째 차원(1번째)의
                # 벡터의 두번째(1번째) 차원에 저장하고

                # "MMM"이 아니거나 메디박 아이디가 아니면 해당유닛은 위와 같은 차원에 무기 쿨타임/맥스쿨타임 을 저장한다

                ally_state[al_id, 2] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y
                # 글로벌 스테이트에 들어가는 정보로 맵 중앙에서의 상대좌표값을 계산한뒤에
                # 글로벌 스테이트 정보의 특정에이전트(for문에 의해 인덱스가 증가하는)의 텐서 두번째 차원(1번째)의
                # 벡터의 세번째(2번째), 네번째(3번째) 차원에 저장한다

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_state[al_id, ind] = (
                        al_unit.shield / max_shield
                    )  # shield
                    ind += 1
                # 아군의 셀프쉴드 비트가 양수이면 글로벌 스테이트 정보의 특정에이전트(for문에 의해 인덱스가 증가하는)의 텐서 두번째 차원(1번째)의
                # 벡터의 다섯번째(4번째, ind=4) 차원에 저장하고 
                # 그다음 정보는 여섯번째 차원(5번째, ind=5)에 저장하도록 ind(인덱스) 값을 증가시킨다

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_state[al_id, ind + type_id] = 1
                # 아군의 셀프쉴드 비트가 0이면 유닛타입id를
                # 글로벌 스테이트 정보의 특정에이전트(for문에 의해 인덱스가 증가하는)의 텐서 두번째 차원(1번째)의
                # 벡터의 ind 차원에 저장한다. (shield_bits에 따라 ind(인덱스) 값이 달라지도록 함) 

        for e_id, e_unit in self.enemies.items():
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (
                    e_unit.health / e_unit.health_max
                )  # health
                enemy_state[e_id, 1] = (
                    x - center_x
                ) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (
                    y - center_y
                ) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_state[e_id, ind] = (
                        e_unit.shield / max_shield
                    )  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_state[e_id, ind + type_id] = 1
            ## 위의 아군 에이전트와 비슷한 메커니즘으로 글로벌 스테이트 정보를 넘겨준다 두번째(1번째) 차원인 유닛쿨다운 정보만 빠져있고
            ## 나머지는 한차원씩 앞으로 밀린다고 보면된다.

        state = np.append(ally_state.flatten(), enemy_state.flatten())
        if self.state_last_action:
            state = np.append(state, self.last_action.flatten())
        if self.state_timestep_number:
            state = np.append(state,
                              self._episode_steps / self.episode_limit)

        state = state.astype(dtype=np.float32)
        ## state정보를 아군에 대한 정보와 적군에 대한 정보를 각각 flatten 하고 마지막 행동과 타임스텝번호를 append 해준다
        ## state_last_action은 기본이 true이고, timestep_number는 false 이다
        ## 다만 여기서 아[적]군 에이전트 정보(에이전트1체력좌표~~,에이전트2체력좌표), 마지막 행동(에이전트1,2,3), 타임스탭정보를 모두 flatten 해서 넣어줘도 상관이없는지 궁금하다
        ## 특히 transfer 러닝할때 글로벌 상태정보의 차원별로 가지는 정보가 에이전트의 쉴드 유무등에 따라 달라질 수 있어서
        ## 이부분은 좀더 생각해봐야 될것같다


        if self.debug:
            logging.debug("STATE".center(60, "-"))
            logging.debug("Ally state {}".format(ally_state))
            logging.debug("Enemy state {}".format(enemy_state))
            if self.state_last_action:
                logging.debug("Last actions {}".format(self.last_action))

        return state
        ## 디버그가 true 로 설정되어있으면 해당정보를 출력시켜준다

    def get_obs_enemy_feats_size(self):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        if self.obs_terrain_height:
            nf_en += 1

        return self.n_enemies, nf_en
        ## 적군 피쳐 사이즈를 반환하는 메소드 이다


    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.obs_last_action:
            nf_al += self.n_actions

        if self.obs_terrain_height:
            nf_al += 1

        return self.n_agents - 1, nf_al
        ## 아군 피쳐 사이즈를 반환하는 메소드 이다


    # custom code
    def get_obs_neutral_feats_size(self):
        """Returns the dimensions of the matrix containing neutral entity features.
        Size is n_neutral x n_features.
        """
        nf_neut = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_neut += 1 + self.shield_bits_enemy

        if self.obs_terrain_height:
            nf_neut += 1

        return self.n_neutrals, nf_neut



    def get_obs_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features.
        """
        # Add 2 to get information of (x,y) codination (increase feature size)
        own_feats = self.unit_type_bits + 2
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1
        if self.obs_terrain_height:
            own_feats += 1


        return own_feats
        ## 자신의 피쳐사이즈를 반환하는 메소드이다

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats
        ## 이동관련 피쳐 사이즈 이다.

    def get_obs_size(self):
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()
        n_neutrals, n_neutral_feats = self.get_obs_neutral_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats
        neutral_feats = n_neutrals * n_neutral_feats

        return move_feats + enemy_feats + ally_feats + own_feats + neutral_feats
        ## observation 사이즈를 반환하는 메소드이다

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

        ## 글로벌 스테이트 사이즈를 반환하는 메소드이다

    def get_visibility_matrix(self):
        """Returns a boolean numpy array of dimensions 
        (n_agents, n_agents + n_targets) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_agents, self.n_agents + self.n_targets),
            dtype=np.bool,
        )

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if (dist < sight_range and e_unit.health > 0):
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # custom code
                # neutral unit
                for neut_id, neut_unit in self.neutrals.items():
                    neut_x = neut_unit.pos.x
                    neut_y = neut_unit.pos.y
                    dist = self.distance(x, y, neut_x, neut_y)

                    if (dist < sight_range and neut_unit.health > 0):
                        # visible and alive
                        arr[agent_id, self.n_agents + self.n_enemies + neut_id] = 1



                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_agents)
                    if al_id > agent_id
                ]
                for i, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if (dist < sight_range and al_unit.health > 0):
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1
                arr[agent_id, agent_id] = 1

        return arr
        ## visibility 메트릭스로 각에이전트에 대하여 포문이 돌면서 적군 유닛 및 아군에이전트와의 거리를 계산해서 
        ## 관측가능한지와 살아있는지 여부에 따라 메트릭스의 해당 인덱스에 True값을 저장한다

    def get_communication_range(self, unit_a, unit_b):
        if unit_a == 919 or unit_b == 918:
            return 16
        elif unit_b == 919 or unit_a == 918:
            return 16
        else:
            return 12
        


    def set_visibility_matrix(self, obs_broadcast_info, obs_communicate_info):
        """Returns a boolean numpy array of dimensions 
        (n_agents, n_agents + n_targets) indicating which units
        are visible to each agent.
        """
        arr = np.zeros(
            (self.n_agents, self.n_agents + self.n_targets),
            dtype=np.bool,
        )

        for agent_id in range(self.n_agents):
            current_agent = self.get_unit_by_id(agent_id)
            if current_agent.health > 0:  # it agent not dead
                x = current_agent.pos.x
                y = current_agent.pos.y
                sight_range = self.unit_sight_range(agent_id)

                # Enemies
                for e_id, e_unit in self.enemies.items():
                    e_x = e_unit.pos.x
                    e_y = e_unit.pos.y
                    dist = self.distance(x, y, e_x, e_y)

                    if (dist < sight_range and e_unit.health > 0):
                        # visible and alive
                        arr[agent_id, self.n_agents + e_id] = 1

                # custom code
                # neutral unit
                for neut_id, neut_unit in self.neutrals.items():
                    neut_x = neut_unit.pos.x
                    neut_y = neut_unit.pos.y
                    dist = self.distance(x, y, neut_x, neut_y)

                    if (dist < sight_range and neut_unit.health > 0):
                        # visible and alive
                        arr[agent_id, self.n_agents + self.n_enemies + neut_id] = 1



                # The matrix for allies is filled symmetrically
                al_ids = [
                    al_id for al_id in range(self.n_agents)
                    if al_id > agent_id
                ]
                for i, al_id in enumerate(al_ids):
                    al_unit = self.get_unit_by_id(al_id)
                    al_x = al_unit.pos.x
                    al_y = al_unit.pos.y
                    dist = self.distance(x, y, al_x, al_y)

                    if (dist < sight_range and al_unit.health > 0):
                        # visible and alive
                        arr[agent_id, al_id] = arr[al_id, agent_id] = 1
                arr[agent_id, agent_id] = 1
            # set_arr_end (similar to visibility_matrix)



        output_matrix = arr.copy()

        if (obs_broadcast_info == True and obs_communicate_info == False):
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i > j:
                        agent_i = self.get_unit_by_id(i)
                        agent_j = self.get_unit_by_id(j)
                        if agent_i.health > 0 and agent_j.health > 0:
                            for k in range(self.n_agents+self.n_targets):
                                if output_matrix[j][k] or output_matrix[i][k] == 1:
                                    output_matrix[j][k] = 1
                                    output_matrix[i][k] = 1
                        
            self.visibility_matrix = output_matrix


        elif (obs_broadcast_info == False and obs_communicate_info == True):

            com_arr = np.zeros(
                (self.n_agents, self.n_agents),
                dtype=np.bool,
            )

            for agent_id in range(self.n_agents):
                current_agent = self.get_unit_by_id(agent_id)
                if current_agent.health > 0:  # it agent not dead
                    x = current_agent.pos.x
                    y = current_agent.pos.y
                    cur_type = current_agent.unit_type 
                    sight_range = self.unit_sight_range(agent_id)

                    al_ids = [
                        al_id for al_id in range(self.n_agents)
                        if al_id > agent_id
                    ]
                    for i, al_id in enumerate(al_ids):
                        al_unit = self.get_unit_by_id(al_id)
                        al_x = al_unit.pos.x
                        al_y = al_unit.pos.y
                        al_type = al_unit.unit_type
                        dist = self.distance(x, y, al_x, al_y)
                        com_range = self.get_communication_range(cur_type, al_type)


                        if (dist < com_range and al_unit.health > 0):
                            # in_com_range and alive
                            com_arr[agent_id, al_id] = com_arr[al_id, agent_id] = 1
                    com_arr[agent_id, agent_id] = 1

            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if i > j:
                        if com_arr[i][j] == 1:
                            for k in range(self.n_agents+self.n_targets):
                                if arr[j][k] or arr[i][k] == 1:
                                    output_matrix[j][k] = 1
                                    output_matrix[i][k] = 1
                        
            self.visibility_matrix = output_matrix


        else:
            self.visibility_matrix = output_matrix







    # 에이전트가 obs를 관측할때 쓰이며 아군이든 본인자신이든 (ally feature or own feature) 형식매개변수 ally에는 True 값이 전달되어서 사용된다.
    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if self.map_type == "plain_neut_tank":
                if unit.unit_type == 3225: # marine_RL
                    type_id = 0
                elif unit.unit_type == 918: # Siegetank_sieged_RL
                    type_id = 1
                elif unit.unit_type == 919: # Siegetankmode_RL
                    type_id = 2
                elif unit.unit_type == 3224: # marauder_RL
                    type_id = 3
                else:
                    type_id = 4 # tank(armored)_RL
         # "marines"맵의 경우 아군인 marine_RL이 3580 이므로 아까 리턴했던 self._min_unit_type이 3580이다.
         # 또한 해당맵에서 marine_RL의 경우 3580이므로 결론적으로 마린은 type_id가 0이 된다.
         # 에이전트 기준 아군이거나 본인 자신은 실제 유닛타입에서 가장 낮은 유닛타입으로 빼준것을 사용한다.
 
 
         # 아래는 @@@@적군@@@@ 유닛타입으로 지도별로 0부터 순차적으로 배정한다.(정확한 기준은 이해안감)
        else:  # use default SC2 unit types
            if self.map_type == "stalkers_and_zealots":
                # id(Stalker) = 74, id(Zealot) = 73
                type_id = unit.unit_type - 73
            elif self.map_type == "colossi_stalkers_zealots":
                # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                if unit.unit_type == 4:
                    type_id = 0
                elif unit.unit_type == 74:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "bane":
                if unit.unit_type == 9:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "MMM":
                if unit.unit_type == 51:
                    type_id = 0
                elif unit.unit_type == 48:
                    type_id = 1
                else:
                    type_id = 2
            elif self.map_type == "marines_ghost":
                if unit.unit_type == 48:
                    type_id = 0
                else:
                    type_id = 1
            elif self.map_type == "marines_tank":
                if unit.unit_type == 48:
                    type_id = 0
                else:
                    type_id = 1

            elif self.map_type == "tanks":
                if unit.unit_type == 33:
                    type_id = 0
                else:
                    type_id = 1

            # custom code
            elif self.map_type == "marineswithneutral":
                # 3580 is marine_RL, but this elif below is for enemy and neutral.
                # 48 is 순정마린(적군)
                if unit.unit_type == 3223:
                    type_id = 0
                # 312 is visible rock(바위는 좀 다양함), 1235 is snapshot tree entity
                else:
                    type_id = 1
            elif self.map_type == "tank_withneutral":
                if unit.unit_type == 1235:
                    type_id = 2
                else:
                    type_id = unit.unit_type - 2177
            elif self.map_type == "plain_neut_tank":
                if unit.unit_type == 48: # en_marine
                    type_id = 0
                elif unit.unit_type == 2177: # en_tank_Sieged (artillery)
                    type_id = 1
                elif unit.unit_type == 2178: # en_tank_tankmode (artillery)
                    type_id = 2
                elif unit.unit_type == 51: # en_marauder
                    type_id = 3
                elif unit.unit_type == 3228: # en_armored_tank
                    type_id = 4
                elif unit.unit_type == 1235: # korhal tree
                    type_id = 5
                elif unit.unit_type == 1236: # korhal foliage
                    type_id = 6
                else:
                    type_id = 7 # Rocks


        return type_id
        # 지도에 따라 유닛타입을 부여하는 코드로, 아군의 유닛타입(에이전트 유닛타입은 커스텀지도에서 별도 작성)과
        # sc2 기본설정 타입값이 다르므로 아군인지 아닌지로 구분하여 지도에 따라 타입 id를 0부터 순차적으로 부여한다

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1


            # ability(using skill) should be allowed
            avail_actions[6] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(unit)








            # custom code
            target_items = self.targets.items()
            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [
                    (t_id, t_unit)
                    for (t_id, t_unit) in self.agents.items()
                    if t_unit.unit_type != self.medivac_id
                ]

            visibility_mask = 0
            for t_id, t_unit in target_items:
                visibility_mask = self.visibility_matrix[agent_id][self.n_agents + t_id]
                if visibility_mask == 1 and t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1




#            enemy_target_items = self.enemies.items()
#            if self.map_type == "MMM" and unit.unit_type == self.medivac_id:
#                # Medivacs cannot heal themselves or other flying units
#                enemy_target_items = [
#                    (t_id, t_unit)
#                    for (t_id, t_unit) in self.agents.items()
#                    if t_unit.unit_type != self.medivac_id
#                ]
#
#            for t_id, t_unit in enemy_target_items:
#                if t_unit.health > 0:
#                    dist = self.distance(
#                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
#                    )
#                    if dist <= shoot_range:
#                        avail_actions[t_id + self.n_actions_no_attack] = 1
#
#
#            neutral_target_items = self.neutrals.items()
#
#            for neu_id, neu_unit in neutral_target_items:
#                if neu_unit.health > 0:
#                    dist = self.distance(
#                        unit.pos.x, unit.pos.y, neu_unit.pos.x, neu_unit.pos.y
#                    )
#                    if dist <= shoot_range:
#                        avail_actions[neu_id + self.n_actions_no_attack + self.n_enemies] = 1            


            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)
        ## 가능한 액션을 뽑는 메소드로 기본적으로 살아있는지 죽었는지 여부로 구분하는 if ~ else문이 있으며, 살았을시에는
        ## 이동가능여부와 적이 사거리안에 있는지 여부를 기준으로 공격가능한 에이전트에 해당하는 인덱스에 1을 넣어준다


    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
        ## 에이전트들의 가용액션을 append 해준다


    def close(self):
        """Close StarCraft II."""
        if self._sc2_proc:
            self._sc2_proc.close()
        # sc2 process를 종료하는 메소드이다


    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed
        # 환경에서 사용되는 렌덤시드를 반환한다

    def render(self):
        """Not implemented."""
        pass
        ## ???무슨용도인가

    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0]
        units_alive.extend([unit.tag for unit in self.neutrals.values() if unit.health > 0])
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)
        # 컨트롤러에 디버그 커맨드를 던져서 지도상의 모든유닛을 킬한다

    # 게임 에피소드의 첫단계에서 아군적군 개체에 대한 정보를 초기화 시키고 숫자가 맞는지 체크한 뒤에 
    # 이상없으면 1스텝 진행, 오류가 발생하면 런쳐를 재실행하거나 리셋을 다시한번하는 코드
    def init_units(self):
        """Initialise the units."""
        self.n_ally_alive = deepcopy(self.n_agents)
        
        n_loop = 0
        while True:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}
            self.neutrals = {}
            self.targets = {}
            ## self.agents 와 self.enemies 딕셔너리 타입 변수를 선언


            ally_units = [
                unit
                for unit in self._obs.observation.raw_data.units
                if unit.owner == 1
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )



            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]
                if self.debug:
                    logging.debug(
                        "Unit {} is {}, x = {}, y = {}".format(
                            len(self.agents),
                            self.agents[i].unit_type,
                            self.agents[i].pos.x,
                            self.agents[i].pos.y,
                        )
                    )
 
            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 2 and unit.health == unit.health_max:
                    self.enemies[len(self.enemies)] = unit
                    if self._episode_count == 0:
                        self.max_reward += unit.health_max + unit.shield_max


            n_enemy = 0
            n_enemy = len(self.enemies)
            # custom code
            for unit in self._obs.observation.raw_data.units:
                if unit.owner == 16 and unit.health == unit.health_max:
                    self.neutrals[len(self.neutrals)] = unit

            self.targets.update(self.enemies)
            vals = []
            vals = list(self.neutrals.values())

            for i in range(len(self.neutrals)):
                self.targets[i+n_enemy] = vals[i]


            if self._episode_count == 0:
                min_unit_type = min(
                    unit.unit_type for unit in self.agents.values()
                )
                self._init_ally_unit_types(min_unit_type)

            all_agents_created = (len(self.agents) == self.n_agents)
            all_targets_created = (len(self.enemies) == self.n_enemies) and (len(self.neutrals) == self.n_neutrals)
            
            

            if all_agents_created and all_targets_created:  # all good
                if self._obs.observation.score.score_details.collected_minerals == 0:
                    return

            n_loop += 1
            if n_loop > 0:
                if len(self.enemies) == 0 or len(self.enemies) == 1:
                    self.reset()


            try:
                self._controller.step(1)
                self._obs = self._controller.observe()

            except (protocol.ProtocolError, protocol.ConnectionError):
                self.full_restart()
                self.reset()

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """


        # Store previous state (이전 스테이트의 엔티티에 관한 글로벌 정보를 deepcopy하여 저장하는 객체)
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)
        self.previous_neutral_units = deepcopy(self.neutrals)
        self.targets = {}

        # 각각 아군 적군 중립 생존유닛 숫자를 저장하는 변수
        n_ally_alive = 0
        n_enemy_alive = 0
        n_neutral_alive = 0
        for al_id, al_unit in self.agents.items():
            updated = False
            # 유닛 태그를 비교하여 같은 유닛태그인 에이전트가 있으면 다시 현재 self.agents 딕셔너리에 저장하고
            # updated 변수를 False에서 True 값으로 바꿔줌, 살아있으면 n_ally_alive 변수에 1추가
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    # 태그가 같은게 있으면 해당 al_id 인덱스에 해당 unit 정보를 다시 넣어줌
                    updated = True
                    n_ally_alive += 1
                    break
            if not updated:  # dead
                al_unit.health = 0




        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break
        
            if not updated:
                 e_unit.health = 0

        # Custom
        self.delta_n_killed_unit = 0
        self.previous_killed_value_units = 0
        self.previous_killed_value_units = deepcopy(self.killed_value_units)
        self.killed_value_units = self._obs.observation.score.score_details.collected_minerals


        self.delta_n_killed_unit = (self.killed_value_units - self.previous_killed_value_units)

        # Avoid Sync miss error
        if self.delta_n_killed_unit < 0:
            self.delta_n_killed_unit = 0

        self.n_killed_units += self.delta_n_killed_unit



        # custom code
        for neut_id, neut_unit in self.neutrals.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if neut_unit.tag == unit.tag:
                    self.neutrals[neut_id] = unit
                    updated = True
                    n_neutral_alive += 1
                    break

            if not updated:  # dead
                neut_unit.health = 0
         # if self.n_enemies - self.n_killed_units < 2:
         #     print("kill_units", self.n_killed_units, "epi_ct", self._episode_count, "epi_step",self._episode_steps)


        self.targets.update(self.enemies)
        vals = []
        vals = list(self.neutrals.values())
        for i in range(self.n_neutrals):
            self.targets[i+self.n_enemies] = vals[i]


        if (n_ally_alive == 0 and self.n_killed_units < self.n_enemies
                or self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and self.n_killed_units == self.n_enemies
                or self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and self.n_killed_units == self.n_enemies:
            return 0

        return None



    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1


# for new maps
        elif self.map_type == "marines_marauder":
            self.marine_id = min_unit_type
            self.marauder_id = min_unit_type + 1
        elif self.map_type == "marines_ghost":
            self.marine_id = min_unit_type
            self.ghost_id = min_unit_type + 1
        elif self.map_type == "marines_tank":
            self.siegetanksieged_id = min_unit_type
#            self.siegetanksieged_id = min_unit_type + 1
            self.siegetank_id = min_unit_type + 1
            self.marine_id = min_unit_type + 2
        elif self.map_type == "marines_scv":
            self.marine_id = min_unit_type
            self.scv_id = min_unit_type + 1
        elif self.map_type == "tanks":
            self.siegetank_id = min_unit_type
            self.siegetanksieged_id = min_unit_type + 1
        elif self.map_type == "marineswithneutral":
            self.marine_id = min_unit_type



    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a
                for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a
                for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    def get_unit_by_id(self, a_id):
        """Get unit by ID."""
        return self.agents[a_id]

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats

    def get_env_info(self):
        env_info = super().get_env_info()
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info
