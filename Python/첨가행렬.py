def change_Augmented(m) :
  for i in range(len(m)) :        #위치별 인덱스 만들어주기 (무조건 1로)
    if m[i][i] != 0 :             # 행렬 (i,i) 값이 존재한다면 위치를 바꿔줄 이유가 없으므로
      value = m[i][i]             # (i,i)을 1로 만들기 위해 해당값을 해당 행 전부 나눠준다.
      for j in range(len(m[i])) :
        m[i][j] = m[i][j] / value
        if m[i][j] == 0 :
              m[i][j] = 0.0
    else :                       #(i,i)가 0이라면 다음 행들중에 (i,i)값이 존재하는지 확인
      for j in range(i+1,len(m)) :
        if m[j][i] != 0 :        #(j,i)가 0이 아닐경우 해당 위치 변경 및 1로 만들어주는 작업 동일시행
          t = m[j][:]
          m[j][:] = m[i][:]
          m[i][:] = t[:]
          for k in range(len(m[j])) :
            m[i][k] = m[i][k] / m[i][i]
            if m[j][k] == 0 :
              m[j][k] = 0.0
          break                  # 포문 탈출
    for j in range(i+1,len(m)) : # 해당 i행으로 그 이외에 존재할 i열에 존재할 수 있는 값 모두 0으로 만들어주기
      if m[i][i] == 1 and m[j][i] != 0 : #0이 아닐경우 0으로 만들기 위해 다음과 같이 실행
        value = m[j][i]
        for k in range(len(m[j])) : 
          m[j][k] = m[j][k] - (value * m[i][k])
          if m[j][k] == 0 :
              m[j][k] = 0.0
  return m

#다음 행렬은
# 1 1 1
# 0 1 1 
# 0 0 1 과 같은 느낌으로 만들어짐 이걸 역순으로 해당 열에 있는 값들 전부 빼준다  
def solving_Augmented(m) :          
  for i in range(len(m)-1,-1,-1) :
    if m[i][i] != 0 :
      for j in range(i-1,-1,-1) :
        if m[j][i] != 0 :
          value = m[j][i]
          for k in range(len(m[j])) :
            m[j][k] =  m[j][k] - (value * m[i][k])
            if m[j][k] == 0 :
              m[j][k] = 0.0
  return m

def det(m) : #det 구하는 함수
  total = 0
  if len(m) == 1 : #size 1일경우 그냥 반환
    return m[0][0]
  if len(m) == 2 : #size가 2일경우 다음과 같이 진행
    return m[0][0] *m[1][1] - m[0][1]*m[1][0]
  else : 
    for j in range(len(m)) :
      temp_matrix = []
      for k in range(len(m)) :
        if k == 0: continue         #해당 행을 skip
        temp = []
        for l in range(len(m)) :
          if l == j : continue #해당 열을 skip한다
          temp.append(m[k][l])
        temp_matrix.append(temp) #i행 j열을 제외한 나머지 원소들로 matrix 생성
      if (j%2) == 0 :  
        total = total + m[0][j]*det(temp_matrix) #det값을 구하러 들어가고 total에 저장
      else :
        total = total - m[0][j]*det(temp_matrix)
    return total

while True :     
  r = int(input('행렬 A의 행 크기를 입력하시오 :'))
  c = int(input('행렬 A의 열 크기를 입력하시오 :'))
  if r == c :
    break
  else : 
    print('행과 열 값 같지않음')

A = []

for i in range(r) :
  t = []
  for j in range(c) :
    input_text = 'A[{}][{}] :'.format(i,j) 
    temp = float(input(input_text))
    t.append(temp)
  A.append(t)

print('Matrix A')
for i in range(r) :
  print(A[i])

while True :
  b = int(input('벡터b의 크기를 입력하시오:'))
  if b == r :
    break
  else :
    print('벡터 크기와 행렬 크기를 맞추세요')

vb = []

for i in range(b) :
  t=[]
  text_input = 'b[{}] : '.format(i)
  data = float(input(text_input))
  t.append(data)
  vb.append(t)

for i in range(b) :
  print(vb[i])

plus_A = []
for i in range(r) :
  plus_A.append(A[i]+vb[i])

print('Matrix A + vector b')
for i in range(r) :
  print(plus_A[i])

Augmented = change_Augmented(plus_A)
print('Augmented matrix during solving process')
for i in range(len(Augmented)) : 
  print(Augmented[i])

solving_augmented = solving_Augmented(Augmented)

print('Augmented matrix after solving process')
for i in range(len(solving_augmented)) :
  print(solving_augmented[i])
print('Solution vector x :')
for i in range(len(solving_augmented)) :
  sv = []
  if solving_augmented[i][i] == 1 :
    sv.append(solving_augmented[i][len(solving_augmented[i])-1])
  else :
    sv.append('NAN')
  print(sv)

print()
determination = det(A)
print("Determinant of A: {}".format(determination))