결과 요약
핵심 수치 (N = 1000만)
케이스시간비율이터레이터 1개 (range)796ms1.00x (기준)이터레이터 2개 (zip + range)1140ms1.43x 느림이터레이터 1개 (list)1403ms1.76x이터레이터 2개 (zip + list)2378ms2.99x 느림

왜 느릴까?
zip 호출 오버헤드

zip은 매 루프마다 두 이터레이터에서 각각 __next__()를 호출 → 함수 호출 비용 2배
결과를 튜플로 패킹((x, y)) 후 언패킹하는 비용 추가

메모리 접근 패턴

range는 lazy(값을 계산), list는 실제 메모리 접근
list 2개면 캐시 미스 가능성 증가 → 2.99x까지 벌어짐



게 보입니다.

진짜 원인: tp_iternext 호출 횟수
메모리 접근이 아니라 CPython 인터프리터의 C 함수 디스패치 비용입니다.
zip.__next__ 내부에서 일어나는 일
zip.__next__() 호출 1번
  └─ range_iterator.tp_iternext()  ← C 함수 포인터 호출 #1
  └─ range_iterator.tp_iternext()  ← C 함수 포인터 호출 #2
  └─ PyTuple_New(2)                ← 튜플 힙 할당
  └─ return tuple
one_iter는 루프당 tp_iternext 1번, zip은 2번 + 튜플 생성입니다.
바이트코드로 보면
one_iterzip루프당 핵심 opcodesFOR_ITER → STORE_FASTFOR_ITER → UNPACK_SEQUENCE → STORE_FAST x2C 레벨 iternext1회2회 + PyTuple_New
no_unpack이 오히려 더 느린 이유
tup[0] 인덱스 접근(BINARY_SUBSCR)이 UNPACK_SEQUENCE보다 오히려 비쌉니다. UNPACK_SEQUENCE는 튜플 크기를 알고 있어서 직접 포인터 접근으로 최적화되어 있기 때문입니다.
결론

메모리/캐시 미스 아님 — PyObject는 말씀대로 모두 heap, list든 range든 포인터 따라가는 건 동일
진짜 원인 = C 레벨 tp_iternext 호출 횟수 × 루프 횟수 + PyTuple_New 비용
1000만 루프면 이 누적이 ~40% 차이로 나타나는 것


Python의 유연성 비용 구조
아키텍처적 근본 원인
CPython은 "모든 것이 PyObject"* 라는 단일 추상화를 선택했습니다. 이게 유연성의 원천이자 오버헤드의 원천입니다.
┌─────────────────────────────────────────────────────┐
│  레이어 1: eval loop  (27.7ns/iter, 47%)            │
│  - 매 opcode마다 switch-dispatch                    │
│  - JUMP_BACKWARD마다 eval_breaker 플래그 체크(GIL) │
│  - 3.11+ adaptive specialization으로 일부 완화      │
├─────────────────────────────────────────────────────┤
│  레이어 2: PyObject 타입 디스패치  (31.5ns/iter)    │
│  - ob_type->tp_as_number->nb_add  함수 포인터 2단계 │
│  - INCREF / DECREF (refcount) — 모든 객체 접근마다  │
│  - small int cache (-5~256) 로 일부 할당 회피       │
├─────────────────────────────────────────────────────┤
│  레이어 3: zip 추가 비용 (~+40%)                    │
│  - tp_iternext 2회 (함수 포인터 호출 × N)           │
│  - PyTuple_New(2) + INCREF×2 + DECREF×2 매 루프     │
│  - UNPACK_SEQUENCE opcode                           │
└─────────────────────────────────────────────────────┘


측정 결과 핵심
비교결과eval loop만 (빈 루프)루프 비용의 47% 가 연산 전 이미 소비float이 int보다 빠름CPython int는 arbitrary precision (ob_digit[]), float은 C double 래핑이라 단순polymorphic 타입 혼합specialization 캐시 미스 → ~6% 추가 손실numpy vs CPython90배 차이 — PyObject 레이어 완전 우회 시
설계 트레이드오프
CPython이 이 비용을 의도적으로 감수한 이유:

int에 메서드 추가, + 연산자 오버라이드, 런타임 타입 교체 — 모두 ob_type 포인터 하나로 가능
함수 포인터 디스패치 구조가 없으면 이런 동적 다형성 자체가 불가능
3.11의 adaptive specialization은 "반복된 타입 패턴을 감지해서 포인터 디스패치를 인라인으로 교체" 하는 방식으로 이 비용을 런타임에 학습해서 줄이는 시도

결국 zip의 ~40% 오버헤드는 메모리 문제가 아니라, N번의 C 함수 포인터 호출 + N번의 힙 할당이 그냥 누적된 겁니다.



아요, perf + objdump로 실제 ASM 레벨까지 파봅시다.명령 20개 실행함명령 20개 실행함
callgrind 실측 결과 (ASM 수준 분석)
총 명령어 수 비교 (N=500,000)
one_itertwo_iter (zip)비율총 명령어 수 (Ir)629,865,7161,057,627,8451.68x총 읽기 (Dr)177,831,807301,510,1911.70x총 쓰기 (Dw)91,460,000156,624,9821.71xLL cache miss57,19657,242≈ 동일
캐시 미스가 거의 동일 → 처음 가설대로 메모리/캐시 문제 아님이 완전히 확인됨.

함수별 명령어 수 (핫패스 분해)
함수one_iter (Ir)two_iter (Ir)증가량_PyEval_EvalFrameDefault53M86M+33MPyObject_Malloc34M68M+34M ← 2배PyObject_Free50M98M+48M ← 2배PyLong_FromLong18M37M+19M ← 2배_PyLong_Add11M28M+17Mlong_dealloc8M13M+5Mzip_next021M+21M ← 신규rangeiter_next(eval loop 내 inlined)12M별도 함수 호출로 분리됨

ASM으로 본 핵심 차이
range_iternext (14 instructions, tail-call)
asm589b00: endbr64
589b04: mov  0x20(%rdi),%rax    ; length_remaining 로드
589b08: test %rax,%rax
589b0b: jle  589b30             ; 끝이면 NULL 리턴
589b0d: mov  0x10(%rdi),%rdx    ; current value
589b11: mov  0x18(%rdi),%rcx    ; step
589b15: sub  $0x1,%rax          ; length--
589b19: mov  %rax,0x20(%rdi)    ; 저장
589b1d: add  %rdx,%rcx          ; next = cur + step
589b20: mov  %rcx,0x10(%rdi)    ; 저장
589b24: mov  %rdx,%rdi
589b27: jmp  PyLong_FromLong    ; ← tail-call로 박싱
zip_next가 루프당 하는 일 (21M instructions)
asm; ① refcount 체크 및 INCREF
5d4990: mov  (%rbx),%r14        ; ob_refcnt 로드
5d4996: cmp  $0x1,%r14          ; immortal 체크
5d499a: jne  ...

; ② 첫 번째 iternext — 함수 포인터 간접 호출
5d49bb: mov  0x18(%rax),%rdi    ; iter[0] 로드
5d49bf: mov  0x8(%rdi),%rax     ; ob_type
5d49c3: call *0xe0(%rax)        ; ← tp_iternext 간접 호출 #1

; ③ NULL 체크 후 DECREF (이전 값)
5d49da: mov  (%rdi),%rax
5d49df: js   ...                ; immortal이면 skip
5d49e1: sub  $0x1,%rax          ; refcnt--
5d49e5: mov  %rax,(%rdi)
5d49e8: je   ...                ; 0이면 dealloc

; ④ 두 번째 iternext — 함수 포인터 간접 호출
5d49fd: mov  0x20(%rax),%rdi    ; iter[1] 로드
5d4a01: mov  0x8(%rdi),%rax     ; ob_type
5d4a05: call *0xe0(%rax)        ; ← tp_iternext 간접 호출 #2

; ⑤ 두 번째 값도 DECREF
; ⑥ PyTuple_New(2) → PyObject_Malloc
; ⑦ 튜플에 두 값 넣고 INCREF×2

결론: 오버헤드의 정체
zip 추가 비용 (+427M instructions) 분해:
  ├─ zip_next 함수 자체:             +21M  (5%)
  ├─ PyObject_Malloc/Free (tuple):   +82M  (19%)  ← 가장 큼
  ├─ PyLong_FromLong (range 2배):    +19M  (4%)
  ├─ eval loop (UNPACK_SEQUENCE 등): +33M  (8%)
  └─ INCREF/DECREF 체인:             나머지 전부
call *0xe0(%rax) — 이 간접 호출 패턴이 핵심입니다. ob_type 포인터를 따라가서 함수 포인터 테이블(tp_as_number, tp_iternext 등)을 다시 따라가는 2단계 포인터 디레퍼런스 + 간접 분기가 루프 N번 × 2회 반복됩니다. CPU의 branch predictor가 간접 호출 대상을 예측하긴 하지만, 이 디스패치 구조 자체를 없앨 수는 없어요. CPython이 "모든 타입은 런타임에 교체 가능"이라는 유연성을 위해 지불하는 고정 세금입니다