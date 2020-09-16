# 2020-1 DCCP Assignment 4
# OMOK.py skeleton code
# - 아래 skeleton code를 기반으로 문제 조건을 만족할 수 있도록 코드를 작성하세요.
# - 필요에 따라 member function을 추가하거나, 주어진 member function의 인자 및 return 값을 변경할 수 있습니다.
#   또는 필요하지 않다면 주어진 skeleton code의 member function을 사용하지 않을 수 있습니다.
# - 본인이 작성한 code의 동작을 설명하는 comment를 자세히 작성하세요.
import random


class OMOK():
    def __init__(self):
        self.width = 10
        self.height = 10
        self.grid = []
        self.record = []        # 게임 진행 상황을 추적하기 위한 변수. 프로그램 구현에 따라 다른 방식으로 기록할 수 있음.

    def init_grid(self):
        self.grid = [
            ['+' for x in range(0, self.width)]
            for y in range(0, self.height)
        ] # 바둑 판 초기화 전체 '+'로 초기화한다
    def init_record(self):
        self.record = [
            [0 for x in range(0, self.width)]
            for y in range(0, self.height)
        ] # 해당 위치에 몇번째로 기록이 되었는지 기록하는 부분, 이는 0으로 전부 초기화 해 놓는다

    # TODO: print_grind
    def print_grid(self): #바둑판 출력
        for i in range(10):
            for j in range(10):
                print(self.grid[i][j], end=' ') # end = ' '을 붙여 한칸 띄워놓고 출력구현
            print()

    # TODO: save
    def save(self): #파일 저장함수
        save_filename = input('[Type the name of replay file] : ') #저장할 파일 명을 적는다.
        put_list = []
        #기록된 돌들의 위치와, 기록 시점을 put_list에 넣는다
        for i in range(10):
            for j in range(10):
                if self.record[i][j] > 0:
                    put_list.append(["{},{}".format(j, i), self.record[i][j]]) 
        sorted(put_list, key=lambda x: x[1]) #기록시점을 가지고 오름차순 정렬
        with open(colab+save_filename, 'w') as f: #파일 열어서 쓴다.
            for i in range(len(put_list)):
                f.write(put_list[i][0] + '\n')
        pass

    # TODO: replay
    def replay(self):
        count = 1
        while True:
            replay_filename = input('[Type the name of replay file or EXIT] : ') # 파일 이름 입력
            try :
                f = open(colab+replay_filename,'r') #파일 열기
                while True:
                    line = f.readline() #라인 읽기
                    if not line: # 라인이 없을 경우 파일을 다 읽었으므로 break
                        break
                    else:           
                        pos = line.split(',') # x,y형식으로 적혀있기 때문에 ,로 다시 split해준다
                        #count로 턴을 알고 있으므로 턴에 따라 진행
                        if count %2 == 0 :
                            if self.ai_turn([int(pos[0]), int(pos[1])]) == True:break 
                        else :
                            if self.user_turn([int(pos[0]), int(pos[1])]) == True:break
                        self.record[int(pos[1])][int(pos[0])] = count
                        count = count + 1
                break
            except OSError: #파일 못열시, 다음과 같이 출력
                print('Replay file does not exist')


    # TODO: menu
    def menu(self):
        while True :
            while True:
                self.init_grid() #바둑판 초기화
                self.init_record() #기록지 초기화
                print('[Menu] (1) Start (2) Replay (3) End ')
                inp = int(input())
                if inp >= 1 and inp <= 3:
                    break
                else:
                    print('[Error - Please Type 1~3]')
            if inp == 1:
                self.play()                        #   1:          오목 게임 실행 (self.play())
            elif inp == 2:                         #   2:          replay 실행 (self.replay())
                self.replay()
            else:
                print('Thanks for using.Bye Bye~')
                break
        pass

    # TODO: play
    def play(self):
        count = 1
        while True:
            print('[Type the location (x,y) of your stone or EXIT or SAVE : ')
            pos = input()
            if pos == 'EXIT' or pos == 'exit': # exit일시 게임 나감
                break
            elif pos == 'SAVE' or pos == 'save': # save로 들어올시 save함수 진행
                self.save()
                break
            else:
                pos = re.sub(' ', '', pos)                           # 2,3 2, 3 같이 들어온 것들을 전부 붙이는 용도
                pos = re.sub(',', ' ', pos)                          # ,를 기준으로 구분
                position = pos.split()                               # split을 사용해 배열화한다 -> x,y이기 때문에 무조건 두개가 있어야함
                if len(position) == 2:                               # 그래서 두개일 때만 조건 실행
                    if 0 <= int(position[0]) and int(position[0]) <= 9 and 0 <= int(position[1]) and int(position[1]) <= 9: #바둑판 안의 위치를 입력하였는지 판단
                        # 해당 위치에 존재하는지 확인하는 함수 넣기
                        if self.exists([int(position[0]), int(position[1])]):  #해당 위치에 돌이 올려져 있는지 판단
                            if count %2 == 0 : # count를 통해 ai의 턴인지 user의 턴인지 판단
                                if self.ai_turn([int(position[0]), int(position[1])]) == True : #짝수면, ai
                                    break
                            else :
                                if self.user_turn([int(position[0]), int(position[1])]) == True : # 홀수면, user
                                    break
                            self.record[int(position[1])][int(position[0])] = count
                            count = count + 1
                        else:
                            print('That location is already occupied') #바둑판에 돌이 위치했을 때, 다음 에러문 출력
                    else:
                        print('There are errors in your input') # 바둑판 규격외의 위치를 입력했을 때 출력
                else:
                    print('There are errors in your input') #두 개로 나누어지지 않았을 경우 출력 
        pass

    # TODO: user_turn 
    def user_turn(self,values):
        print('user add a stone at ({},{})'.format(values[0],values[1])) 
        self.grid[values[1]][values[0]] = 'O' # 사용자의 돌 O를 위치시킨다.
        self.print_grid()                     # 바둑판 출력한다
        if self.is_over(values[1], values[0]) == True : #게임이 끝났는지 물어보고 True일시 게임 종료 선언
            self.game_over('O')
            return True                                 # 끝난걸 알리기 위해 True 반환
        else :
            return False
        pass

    # TODO: ai_turn
    def ai_turn(self,values):
        print('Computer add a stone at ({},{})'.format(values[0],values[1]))
        self.grid[values[1]][values[0]] = 'X'          # 컴퓨터 돌 X를 위치시킨다.
        self.print_grid()                              # 바둑판 출력한다
        if self.is_over(values[1], values[0]) == True :# 게임이 끝났는지 물어보고 True일시 게임 종료 선언
            self.game_over('O')
            return True                                # 끝난걸 알리기 위해 True 반환
        else :
            return False
        pass

    # TODO: is_over 게임이 끝났는지 확인하는 곳
    def is_over(self, x, y):
        c = 1
        lc = 1
        mode = self.grid[x][y]
        #대각선
        while c < 1 : # 바둑판 최대 크기 -> 대각선 방향 확인
            if x+c >= 10 or y+c >= 10 :
                break
            else :
                if mode == self.grid[x+c][y+c] :
                    lc = lc+1
                    c = c+1
                else :
                    break
        c = 1
        while c < 10 :
            if x-c < 0 or y-c < 0 :
                break
            else :
                if mode == self.grid[x-c][y-c] :
                    lc = lc+1
                    c = c+1
                else :
                    break

        #대각선 역방향
        rc = 1
        c= 1
        while c < 10:  # 바둑판 최대 크기 -> 대각선 역방향 확인
            if x + c >= 10 or y - c < 0:
                break
            else:
                if mode == self.grid[x + c][y - c]:
                    rc = rc + 1
                    c = c + 1
                else:
                    break
        c = 1
        while c < 10:  # 바둑판 최대 크기
            if x - c < 0 or y + c < 0:
                break
            else:
                if mode == self.grid[x - c][y + c]:
                    rc = rc + 1
                    c = c + 1
                else:
                    break
        # 가로
        lr = 1
        c = 1
        while c < 10:  # 바둑판 최대 크기 -> 가로줄 확인
            if x + c >= 10:
                break
            else:
                if mode == self.grid[x + c][y]:
                    lr = lr + 1
                    c = c + 1
                else:
                    break
        c = 1
        while c < 10: 
            if x - c < 0:
                break
            else:
                if mode == self.grid[x - c][y]:
                    lr = lr + 1
                    c = c + 1
                else:
                    break
        ud= 1
        c = 1
        while c < 10:  # 바둑판 최대 크기 -> 세로줄 확인
            if y + c >= 10:
                break
            else:
                if mode == self.grid[x][y+c]:
                    ud = ud + 1
                    c = c + 1
                else:
                    break
        c = 1
        while c < 10:
            if y - c < 0:
                break
            else:
                if mode == self.grid[x][y-c]:
                    ud = ud + 1
                    c = c + 1
                else:
                    break
        if ud == 5 or lr == 5 or lc == 5 or rc == 5 :
            return True
        else :
            return False
        pass

    # TODO: game_over
    def game_over(self, user_win): #게임이 끝났을 시, O와 X로 user와 computer를 구분 , 출력진행
        if user_win == 'O' :
            print('[Win : you win!]')
        else :
            print('[DEFEAT : You lose the game]')
        pass

    def exists(self,values): #해당 위치에 이미 돌이 위치해있는지 확인하고 true false 반환
        if self.record[values[1]][values[0]] == 0: #없으면 true 반환
            return True
        else:
            return False

# 아래에 main 함수 작성 (main 함수 외부에 코드 작성 금지)
# hint: 과제 기술 및 skeleton code 양식을 따를 경우, OMOK instance의 menu() 함수를 호출해 게임을 시작.
if __name__ == '__main__':
    omok = OMOK()
    omok.menu()
    pass
