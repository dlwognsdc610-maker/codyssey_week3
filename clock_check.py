from enum import Enum
from typing import List, Dict, Optional, Callable
from collections import deque

"""
 @docs: 
 Simple Toy NPU Simulator
 Impl reason:
  * Simulate the NPU
    -  Simplify Complex Framework
       like ggml, onnx runtime, torch, jax, tf... many ML Frameworks.
    - think it can be usefull reference other 

 * Limits:
     - Virtual Latency Cost 
     - Power
     - Thermal modeling
     - Clock Tree Sync
     - Simplify the architecture

 *HW Structure:
     Referenced Architecture arm Ethos-u85 

   Memory{
       - DRAM
       - DMA_Engine
       - SRAM (ScratchPad. in my reference device contain the execute in Compiler.)
       - Banked Sram       
   } 

   Core{
      - Mac {
        - Multiplier  
        - Accumulator 
      }
      
      SystolicArray {
        - MacUnit[SPEC_MAC]
      }

      - ProgrammableLayerEngine[SPEC_PLE]

      - ScalarAlu[SPEC_SALU]

   }
   ControlFlowEngine{
    - CommandStream
    - Fetch
    - Decode
    - ScoreBoard
    - Dispatch
    - ExecutePipeline
   }

   IR_Graph {  
       - Node {
           - Input Tensor
           - Output Tensor
           - Operation: 
       }
       
       Tensor {
           - DefNode:
           - Consumer:  
       }

   }

   Compiler{
      InstrSet
      ScoreBoardScheduler
      Isel
      Type
      Validator
      PassManager
   }

"""


class TensorType(Enum):
    INT4 = 0
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT64 = 4
    FLOAT16 = 5
    FLOAT32 = 6
    FLOAT64 = 7

class Op(Enum):
    NONE = 0
    ADD = 1
    MUL = 2
    SUB = 3
    DIV = 4
    GEMM = 5
    GEMV = 6
    CONV1D = 7
    CONV2D = 8 

class InstrSet(Enum):
    IAdd4 = 0
    ISub4 = 1
    IMul4 = 2
    IMAC4 = 3
    FAdd32 = 4
    FSub32 = 5
    FMul32 = 6
    FMAC32 = 7
    FAdd64 = 8
    FSub64 = 9
    FMul64 = 10
    FMAC64 = 11

class CommandStream(Enum):
    pass
#HW_Spec
class SpecTable:
    # Constants
    MAX_SRAM_SIZE = 1024 * 512 # 512KB Scratchpad
    VECTOR_LANES = 16          # SIMD width for PLE
    MAC_PER_CYCLE = 256        # 256 MACs for INT8
    # Throughput 
    THROUGHPUT = {
        TensorType.INT4: 2.0,
        TensorType.INT8: 1.0,
        TensorType.INT16: 0.5,
        TensorType.FLOAT32: 0.1 
    }
    # DMA Bandwidth (Bytes/Cycle)
    DMA_BW = 16

class HardwareConfig:
    def __init__(self):
        # Memory Hierarchy
        self.SRAM_BANKS = 4
        # 128KB per bank
        self.BANK_SIZE = 128 * 1024  
        self.SRAM_TOTAL_SIZE = self.SRAM_BANKS * self.BANK_SIZE
        # Core
        self.MCE_UNITS = 256          
        self.PLE_LANES = 16
        # Pipeline: Fetch, Decode, Rename, Execute, Writeback     
        self.PIPELINE_STAGES = 5
        self.MAX_INFLIGHT_INSTR = 32

class Instruction:
    def __init__(self, opcode: Op, src1: int, src2: int, dest: int, t_type: TensorType):
        #SSA ID
        self.opcode = opcode
        self.src1 = src1   
        self.src2 = src2
        self.dest = dest
        self.t_type = t_type

    def __repr__(self):
        return f"[{self.opcode.name} | S1:{self.src1} S2:{self.src2} -> D:{self.dest}]"

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.instructions: List[Instruction] = []
        self.next_block: Optional['Block'] = None

    def add_instr(self, instr: Instruction):
        self.instructions.append(instr)

class CostCycle:
    MAC_UNITS = 256         
    SRAM_BW_PER_CYCLE = 16  
    WEIGHT_DECOMP_RATIO = 2.0


    @staticmethod
    def get_dma_cost(bytes_count: int, is_weight: bool) -> int:
        # Weights take longer due to decompression overhead
        overhead = 20 if is_weight else 10
        return overhead + (bytes_count // CostCycle.SRAM_BW_PER_CYCLE)

    @staticmethod
    def get_mce_cost(m: int, n: int, k: int) -> int:
        total_ops = m * n * k
        return max(1, total_ops // CostCycle.MAC_UNITS)


 
#Access Unit
class BankedSRAM:
    #Busy Table
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.busy_until = [0] * config.SRAM_BANKS

    def access(self, address: int, current_cycle: int) -> bool:
        bank_id = (address // 64) % self.config.SRAM_BANKS # 64-byte interleaving
        if self.busy_until[bank_id] > current_cycle:
            return False #  Stall
        
        self.busy_until[bank_id] = current_cycle + 1
        return True

class BusArbiter:
    def __init__(self, sram: BankedSRAM):
        self.sram = sram
        self.bank_locks = {} # {bank_id: owner_unit_name}

    def request_access(self, unit_name: str, address: int, cycle: int) -> bool:
        bank_id = (address // 64) % self.sram.config.SRAM_BANKS
        
        # Check if bank is already taken this cycle 
        if bank_id in self.bank_locks and self.bank_locks[bank_id] != unit_name:
            return False # STALL 
            
        # Check if hardware bank is physically busy from long access
        if not self.sram.access(address, cycle):
            return False
            
        self.bank_locks[bank_id] = unit_name
        return True

    def clear_locks(self):
        self.bank_locks.clear()

class ScratchPad:
    def __init__(self, size):
        self.memory = [0] * size

    def read(self, addr):
        return self.memory[addr]

    def write(self, addr, value):
        self.memory[addr] = value

class DMA:
    def __init__(self, sram: ScratchPad):
        self.sram = sram
        #assume host memory
        self.dram = {}
    
    def copy_to_sram(self, dram_addr: int, sram_addr: int, size: int, is_weight: bool ) -> int:
        for i in range(size):
            val = self.dram.get(dram_addr + i, 0.0)
            self.sram.write(sram_addr + i, val)
        return CostCycle.get_dma_cost(size, is_weight)+ (size // 2) # Example throughput

    def copy_to_dram(self, sram_addr: int, dram_addr: int, size: int, is_weight: bool) -> int:
        for i in range(size):
            val = self.sram.read(sram_addr + i)
            self.dram[dram_addr + i] = val
        return CostCycle.get_dma_cost(size, is_weight)+ (size // 2)

#Execute Unit
class InternalAccumulator:
    def __init__(self, size=64):
        self.buffer = [0] * size
        self.is_dirty = False

    def write(self, idx, value):
        self.buffer[idx] = value
        self.is_dirty = True

    def read(self, idx):
        return self.buffer[idx]

    def clear(self):
        self.buffer = [0] * len(self.buffer)
        self.is_dirty = False


class Salu:
    """Scalar ALU for control flow and address calculations."""
    def __init__(self, srpc):
        self.srpc = srpc # Scalar Register File

    def execute(self, opcode, rs1, rs2):
        if opcode == InstrSet.IAdd4:
            return rs1 + rs2
        return 0

class Valu:
    """Vector ALU for SIMD operations."""
    def __init__(self, vrpc):
        self.vrpc = vrpc # Vector Register File

    def execute(self, opcode, v1, v2):
        # Implementation for 4-element vector ops
        if opcode == InstrSet.IAdd4:
            return [a + b for a, b in zip(v1, v2)]
        return [0] * 4

class MacUnit:
    """Individual Processing Element (PE) inside the Systolic Array."""
    def __init__(self):
        self.weight = 0
        self.accumulator = 0

    def compute(self, input_val, weight_val):
        self.accumulator += input_val * weight_val
        return input_val

class SystolicArray:
    def __init__(self, mac_size):
        self.mac_size = mac_size
        self.grid = [[MacUnit() for _ in range(mac_size)] for _ in range(mac_size)]

    def pulse(self, inputs, weights):
        # Simplified data shift logic for a single cycle
        for r in range(self.mac_size):
            for c in range(self.mac_size):
                self.grid[r][c].compute(inputs[r], weights[c])

class NPUCore:
    def __init__(self, memory_size=1024):
        self.sram = ScratchPad(memory_size)
        self.salu = Salu(None) # Assuming scalar regs for now
        self.systolic_array = SystolicArray(4)
        self.pc = ProgramCounter()
        self.clock = Clock()

    def step(self, instr: Instruction):
        self.clock.tick()
        
        if instr.opcode == Op.ADD:
            # Simple element-wise simulation
            for i in range(16): # Using size 16 from your example
                val1 = self.sram.read(instr.src1 + i)
                val2 = self.sram.read(instr.src2 + i)
                self.sram.write(instr.dest + i, val1 + val2)
                
        elif instr.opcode == Op.GEMM:
            # This would trigger the Systolic Array 'pulse'
            pass

        self.pc.increment()


#Control Unit
class CommandSplitter:
    def __init__(self):
        self.access_queue = deque()  # DMA / Memory Ops
        self.execute_queue = deque() # MCE / PLE Ops

    def split(self, instructions: List[Instruction]):
        for instr in instructions:
            if instr.opcode in [Op.NONE]: # Logic for LOAD/STORE
                self.access_queue.append(instr)
            else:
                self.execute_queue.append(instr)

class ProgramCounter:
    def __init__(self):
        self.current_pc = 0

    def increment(self):
        self.current_pc += 1

    def jump(self, address):
        self.current_pc = address



class Scoreboard:
    def __init__(self):
        # Maps Tensor ID to the cycle it will be ready
        self.register_ready_cycle = {}
        # Maps Hardware Unit to the cycle it becomes free
        self.unit_busy_until = {
            "MCE": 0,  
            "PLE": 0,  # Programmable Layer Engine (Vector)
            "DMA": 0   
        }

    def is_resource_ready(self, instr: Instruction, current_cycle: int) -> bool:
        # Check Source Dependencies
        src1_ready = self.register_ready_cycle.get(instr.src1, 0) <= current_cycle
        src2_ready = self.register_ready_cycle.get(instr.src2, 0) <= current_cycle
        
        # Check Unit Availability
        unit = self._get_unit_for_op(instr.opcode)
        unit_ready = self.unit_busy_until[unit] <= current_cycle
        
        return src1_ready and src2_ready and unit_ready

    def _get_unit_for_op(self, opcode: Op):
        if opcode == Op.GEMM: return "MCE"
        if opcode == Op.ADD:  return "PLE"
        return "DMA"

    def mark_scheduled(self, instr: Instruction, start_cycle: int, latency: int):
        unit = self._get_unit_for_op(instr.opcode)
        finish_cycle = start_cycle + latency
        
        # Reserve the Unit and the Destination Register
        self.unit_busy_until[unit] = finish_cycle
        self.register_ready_cycle[instr.dest] = finish_cycle

class DAE:
    def __init__(self):
        self.access_queue = deque()  # DMA Tasks
        self.execute_queue = deque() # MCE/PLE Tasks
        self.scoreboard = Scoreboard()
#       self.access_unit
#       self.execute_unit

    def step(self, current_cycle):
        pass
         
        # 1. Access Unit (Runs independently)
  #      if self.access_unit.is_free(current_cycle) and self.access_queue:
   #         task = self.access_queue.popleft()
    #        self.access_unit.dispatch(task, current_cycle)
     #       self.scoreboard.mark_scheduled(task, current_cycle, task.latency)

        # 2. Execute Unit (Stalls only on dependency)
     #   if self.execute_unit.is_free(current_cycle) and self.execute_queue:
      #      task = self.execute_queue[0] 
       #     # Check Scoreboard: Is the data from the Access Unit ready?
        #    if self.scoreboard.is_resource_ready(task, current_cycle):
         #       self.execute_queue.popleft()
          #      self.execute_unit.dispatch(task, current_cycle)
           #    self.scoreboard.mark_scheduled(task, current_cycle, task.latency)

class Semaphore:
    def __init__(self):
        # Maps a semaphore ID to its current value
        self.semaphores = {i: 0 for i in range(16)}

    def signal(self, sem_id: int):
        """Called by a hardware unit when a task is finished."""
        self.semaphores[sem_id] += 1

    def wait(self, sem_id: int, target_val: int) -> bool:
        """Returns True if the dependency is met, False if we must stall."""
        return self.semaphores[sem_id] >= target_val

class SyncManager:
    def __init__(self):
        self.semaphores = {i: 0 for i in range(16)}
        self.waiting_units = {} # {unit_name: (sem_id, target_val)}

    def signal(self, sem_id: int):
        self.semaphores[sem_id] += 1
        print(f"--- Sync: Semaphore {sem_id} incremented to {self.semaphores[sem_id]}")

    def register_wait(self, unit_name: str, sem_id: int, target_val: int):
        self.waiting_units[unit_name] = (sem_id, target_val)

    def is_unit_blocked(self, unit_name: str) -> bool:
        if unit_name not in self.waiting_units:
            return False
        
        sem_id, target = self.waiting_units[unit_name]
        if self.semaphores[sem_id] >= target:
            del self.waiting_units[unit_name] # Requirement met
            return False
        return True # Still waiting

class Dispatcher:
    def __init__(self, sram: ScratchPad):
        self.sram = sram
        self.total_cycles = 0
        self.active_threads = SpecTable.VECTOR_LANES

    def calculate_cost(self, instr: Instruction, size: int) -> int:
        # Scale cost based on TensorType from SpecTable
        base_efficiency = SpecTable.THROUGHPUT.get(instr.t_type, 1.0)
        
        if instr.opcode == Op.GEMM:
            # SIMT-like parallel execution across MAC units
            compute_cycles = (size // SpecTable.MAC_PER_CYCLE) / base_efficiency
            return int(compute_cycles)
        
        if instr.opcode == Op.ADD:
            # Vector/PLE operation
            return int((size // self.active_threads) / base_efficiency)
            
        return 1

    def dispatch_block(self, block: Block):
        """Executes a block of instructions with simplified Tiling logic."""
        for instr in block.instructions:
            # 1. Tiling Check: If size > SRAM, we loop (simplified)
            tensor_size = 1024 
            
            if tensor_size > SpecTable.MAX_SRAM_SIZE:
                num_tiles = (tensor_size // SpecTable.MAX_SRAM_SIZE) + 1
                for _ in range(num_tiles):
                    self.total_cycles += self.calculate_cost(instr, SpecTable.MAX_SRAM_SIZE)
            else:
                self.total_cycles += self.calculate_cost(instr, tensor_size)


class BackScoreboardScheduler:
    def __init__(self, scoreboard: Scoreboard):
        self.scoreboard = scoreboard
        self.issue_queue: List[Instruction] = []
        self.executed_instructions = []

    def add_to_queue(self, instructions: List[Instruction]):
        self.issue_queue.extend(instructions)

    def schedule_cycle(self, current_cycle: int):
        """
        Attempts to dispatch instructions from the queue.
        Implements basic Out-of-Order dispatch.
        """
        dispatched_this_cycle = []
        
        # Scan queue for ready instructions
        for i, instr in enumerate(self.issue_queue):
            if self.scoreboard.is_resource_ready(instr, current_cycle):
                # Calculate Latency (simplified logic)
                latency = 10 if instr.opcode == Op.GEMM else 2
                
                # Update Scoreboard
                self.scoreboard.mark_scheduled(instr, current_cycle, latency)
                
                print(f"Cycle {current_cycle}: Dispatching {instr.opcode.name} to {instr.dest}")
                
                dispatched_this_cycle.append(i)
                self.executed_instructions.append((instr, current_cycle, latency))
                
                # In a limited-issue NPU, we might break here if 
                # we can only issue 1 instr per cycle.
        
        # Remove dispatched instructions from queue (backwards to preserve indices)
        for index in sorted(dispatched_this_cycle, reverse=True):
            self.issue_queue.pop(index)

    def is_done(self):
        return len(self.issue_queue) == 0

class Clock:
    def __init__(self):
        self.cycle = 0

    def tick(self):
        self.cycle += 1

class PipelineState(Enum):
    FETCH = 0
    DECODE = 1
    EXECUTE = 2
    WRITEBACK = 3
    RETIRED = 4

class InFlightInstr:
    def __init__(self, instr: Instruction):
        self.instr = instr
        self.state = PipelineState.FETCH
        self.cycles_remaining = 0
        self.dep_ready = False

class Decoder:
    def decode(self, raw_instr):
        # Returns (Opcode, Dest, Src1, Src2)
        return InstrSet(raw_instr[0]), raw_instr[1], raw_instr[2], raw_instr[3]




#Dual Graph Tensor and Node View 
class Tensor:
    def __init__(self, id: int, size: int, offset: int, t_type: TensorType):
        self.id = id  
        self.size = size
        self.offset = offset
        self.type = t_type

    def __repr__(self):
        return f"Tensor(v{self.id}, size={self.size}, type={self.type.name})"

class Node:
    def __init__(self, op: Op, inputs: List[Tensor], output: Tensor):
        self.op = op
        self.inputs = inputs
        self.output = output


# Memory Context Object
class Ctx:
    def __init__(self, alloc_size: int):
        self.alloc_size = alloc_size
        self.curr_offset = 0
        self.tensor_count = 0
        self.nodes: List[Node] = []

    def allocate(self, size: int, t_type: TensorType) -> Tensor:
        t = Tensor(self.tensor_count, size, self.curr_offset, t_type)
        self.tensor_count += 1
        self.curr_offset += size # Simple linear allocation
        return t



class NpuVirtualDevice:
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.cycle = 0
        self.sram = [0] * config.SRAM_TOTAL_SIZE
        self.scoreboard = Scoreboard()
        self.in_flight = deque()
        self.retired_count = 0

    def tick(self, command_queue: deque):
        self.cycle += 1
        
        # 1. Retirement Stage
        if self.in_flight and self.in_flight[0].state == PipelineState.RETIRED:
            instr = self.in_flight.popleft()
            #self.scoreboard.clear_busy(instr.instr.dest)
            self.retired_count += 1

        # 2. Execution Stage (Check for Latency)
        for op in self.in_flight:
            if op.state == PipelineState.EXECUTE:
                op.cycles_remaining -= 1
                if op.cycles_remaining <= 0:
                    op.state = PipelineState.RETIRED

        # 3. Issue/Decode Stage (Check Scoreboard for Hazards)
        for op in self.in_flight:
            if op.state == PipelineState.DECODE:
                if not self.scoreboard.is_resource_ready(op.instr,self.cycle):
                    op.state = PipelineState.EXECUTE
                    op.cycles_remaining = self.calculate_latency(op.instr)
                    self.scoreboard.mark_scheduled(op.instr.dest, self.cycle, op.cycles_remaining)

        # 4. Fetch Stage
        if len(self.in_flight) < self.config.MAX_INFLIGHT_INSTR and command_queue:
            raw_instr = command_queue.popleft()
            new_op = InFlightInstr(raw_instr)
            new_op.state = PipelineState.DECODE
            self.in_flight.append(new_op)

    def calculate_latency(self, instr: Instruction) -> int:
        # Complex latency logic based on data type and MAC availability
        if instr.opcode == Op.GEMM:
            return max(4, 1024 // self.config.MCE_UNITS)
        return 1

    
class GraphBuilder:
    def __ini__(self, ctx: Ctx):
        self.ctx = ctx
        # Maps user-defined string symbols to the latest SSA Tensor version
        self.symbol_stack: Dict[str, Tensor] = {}

    def set_symbol(self, name: str, tensor: Tensor):
        """Maps a name to an SSA version."""
        self.symbol_stack[name] = tensor

    def get_symbol(self, name: str) -> Tensor:
        if name not in self.symbol_stack:
            raise ValueError(f"Symbol '{name}' not found in SSA stack.")
        return self.symbol_stack[name]

    def emit(self, op: Op, *args: str, out_name: str, size: int, t_type: TensorType) -> Tensor:
        """Core SSA emitter: takes symbol names, produces a new SSA Tensor."""
        input_tensors = [self.get_symbol(arg) for arg in args]
        
        # SSA Requirement: Every assignment creates a new Tensor object (version)
        out_tensor = self.ctx.allocate(size, t_type)
        
        # Record the operation in the graph
        node = Node(op, input_tensors, out_tensor)
        self.ctx.nodes.append(node)
        
        # Update stack to point the symbol name to the newest SSA version
        self.symbol_stack[out_name] = out_tensor
        return out_tensor

    def build_flow(self, flow_fn: Callable[['GraphBuilder'], None]):
        """Executes a lambda or function to stack the symbols."""
        flow_fn(self)


class Compiler:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.executable: List[Instruction] = []

    def compile(self) -> List[Instruction]:
        """Lowers SSA Nodes into NPU Instructions."""
        for node in self.ctx.nodes:
            # Simple 1:1 mapping from Op to Instruction
            # In a real NPU, Op.GEMM might expand into multiple Micro-ops
            
            src1_addr = node.inputs[0].offset
            src2_addr = node.inputs[1].offset if len(node.inputs) > 1 else -1
            dest_addr = node.output.offset
            
            instr = Instruction(
                opcode=node.op,
                src1=src1_addr,
                src2=src2_addr,
                dest=dest_addr,
                t_type=node.output.type
            )
            self.executable.append(instr)
            
        return self.executable


class CFGController:
    def __init__(self):
        self.stack = [] # Divergence Stack

    def handle_branch(self, mask_true: int, mask_false: int):
        """
        Explicitly pushes the 'Else' path onto a stack 
        to be handled after the 'Then' path completes.
        """
        self.stack.append(mask_false)
        return mask_true

    def handle_join(self):
        """Re-converge threads by popping the previous mask."""
        return self.stack.pop()


#class PassManager:
# opt dce
# opt constant prop
# tiling
# unroll
 
    
"""
Test Suite for NPU Simulator
Covers: Memory, Execution Units, Control Flow, IR Graph, Compiler, Scheduling
"""



# ===========================================================================
# 1. Memory Subsystem Tests
# ===========================================================================

class TestScratchPad:
    def test_write_and_read_basic(self):
        sram = ScratchPad(256)
        sram.write(0, 42)
        assert sram.read(0) == 42

    def test_write_multiple_addresses(self):
        sram = ScratchPad(256)
        for i in range(16):
            sram.write(i, i * 10)
        for i in range(16):
            assert sram.read(i) == i * 10

    def test_default_value_is_zero(self):
        sram = ScratchPad(256)
        assert sram.read(100) == 0

    def test_overwrite_existing_value(self):
        sram = ScratchPad(256)
        sram.write(10, 99)
        sram.write(10, 200)
        assert sram.read(10) == 200

    def test_boundary_address(self):
        sram = ScratchPad(256)
        sram.write(255, 7)
        assert sram.read(255) == 7


class TestBankedSRAM:
    def setup_method(self):
        self.config = HardwareConfig()
        self.sram = BankedSRAM(self.config)

    def test_initial_access_succeeds(self):
        assert self.sram.access(0, 0) == True

    def test_same_bank_conflict_stalls(self):
        # First access at cycle 0 marks bank busy until cycle 1
        self.sram.access(0, 0)
        # Second access at same cycle should stall
        assert self.sram.access(0, 0) == False

    def test_access_after_busy_clears(self):
        self.sram.access(0, 0)  # busy_until = 1
        assert self.sram.access(0, 1) == True  # cycle 1 is now free

    def test_different_banks_no_conflict(self):
        # Banks interleave every 64 bytes; address 0 and 64 hit different banks
        self.sram.access(0, 0)
        assert self.sram.access(64, 0) == True

    def test_bank_id_wraps_correctly(self):
        # With 4 banks: address 256 (bank 0) vs address 320 (bank 1)
        result_a = self.sram.access(256, 5)
        result_b = self.sram.access(320, 5)
        assert result_a == True
        assert result_b == True


class TestBusArbiter:
    def setup_method(self):
        config = HardwareConfig()
        self.sram = BankedSRAM(config)
        self.arbiter = BusArbiter(self.sram)

    def test_first_unit_gets_access(self):
        assert self.arbiter.request_access("MCE", 0, 0) == True

    def test_same_unit_can_retry_same_bank(self):
        self.arbiter.request_access("MCE", 0, 0)
        # Same unit, same bank - allowed
        assert self.arbiter.request_access("MCE", 0, 1) == True

    def test_different_units_same_bank_conflict(self):
        self.arbiter.request_access("MCE", 0, 0)
        assert self.arbiter.request_access("PLE", 0, 0) == False

    def test_clear_locks_allows_reaccess(self):
        self.arbiter.request_access("MCE", 0, 0)
        self.arbiter.clear_locks()
        assert self.arbiter.request_access("PLE", 0, 1) == True


class TestDMA:
    def setup_method(self):
        self.sram = ScratchPad(1024)
        self.dma = DMA(self.sram)

    def test_copy_to_sram_returns_positive_cycles(self):
        self.dma.dram[0] = 1.0
        cycles = self.dma.copy_to_sram(0, 100, 1, is_weight=False)
        assert cycles > 0

    def test_copy_to_sram_data_appears_in_sram(self):
        for i in range(8):
            self.dma.dram[i] = float(i)
        self.dma.copy_to_sram(0, 0, 8, is_weight=False)
        for i in range(8):
            assert self.sram.read(i) == float(i)

    def test_copy_to_dram_roundtrip(self):
        self.sram.write(0, 55)
        self.dma.copy_to_dram(0, 500, 1, is_weight=False)
        assert self.dma.dram[500] == 55

    def test_weight_transfer_costs_more_than_activation(self):
        cost_data = self.dma.copy_to_sram(0, 0, 64, is_weight=False)
        cost_weight = self.dma.copy_to_sram(0, 0, 64, is_weight=True)
        assert cost_weight >= cost_data

    def test_missing_dram_address_defaults_to_zero(self):
        self.dma.copy_to_sram(9999, 0, 4, is_weight=False)
        assert self.sram.read(0) == 0.0


# ===========================================================================
# 2. Cost & Spec Tests
# ===========================================================================

class TestCostCycle:
    def test_dma_cost_weight_overhead(self):
        cost_w = CostCycle.get_dma_cost(0, is_weight=True)
        cost_a = CostCycle.get_dma_cost(0, is_weight=False)
        assert cost_w > cost_a

    def test_dma_cost_scales_with_bytes(self):
        small = CostCycle.get_dma_cost(16, is_weight=False)
        large = CostCycle.get_dma_cost(160, is_weight=False)
        assert large > small

    def test_mce_cost_minimum_one(self):
        assert CostCycle.get_mce_cost(1, 1, 1) >= 1

    def test_mce_cost_large_gemm(self):
        cost = CostCycle.get_mce_cost(64, 64, 64)
        expected = max(1, (64 * 64 * 64) // CostCycle.MAC_UNITS)
        assert cost == expected

    def test_mce_cost_scales_with_matrix_size(self):
        small = CostCycle.get_mce_cost(4, 4, 4)
        large = CostCycle.get_mce_cost(32, 32, 32)
        assert large > small


# ===========================================================================
# 3. Execution Unit Tests
# ===========================================================================

class TestInternalAccumulator:
    def test_initial_state_clean(self):
        acc = InternalAccumulator()
        assert acc.is_dirty == False

    def test_write_sets_dirty(self):
        acc = InternalAccumulator()
        acc.write(0, 1.0)
        assert acc.is_dirty == True

    def test_write_and_read(self):
        acc = InternalAccumulator(16)
        acc.write(3, 99.5)
        assert acc.read(3) == 99.5

    def test_clear_resets_state(self):
        acc = InternalAccumulator(16)
        acc.write(0, 7.0)
        acc.clear()
        assert acc.read(0) == 0
        assert acc.is_dirty == False


class TestMacUnit:
    def test_accumulates_correctly(self):
        mac = MacUnit()
        mac.compute(2, 3)  # 2*3 = 6
        mac.compute(4, 5)  # 4*5 = 20 -> total 26
        assert mac.accumulator == 26

    def test_passes_input_through(self):
        mac = MacUnit()
        result = mac.compute(7, 2)
        assert result == 7

    def test_zero_input_no_accumulation(self):
        mac = MacUnit()
        mac.compute(0, 100)
        assert mac.accumulator == 0


class TestSystolicArray:
    def test_grid_dimensions(self):
        sa = SystolicArray(4)
        assert len(sa.grid) == 4
        assert len(sa.grid[0]) == 4

    def test_pulse_does_not_raise(self):
        sa = SystolicArray(4)
        inputs = [1, 2, 3, 4]
        weights = [1, 0, 0, 0]
        sa.pulse(inputs, weights)  # Should not raise

    def test_accumulation_after_pulse(self):
        sa = SystolicArray(2)
        sa.pulse([1, 2], [3, 4])
        # grid[0][0]: input=1, weight=3 -> acc = 3
        assert sa.grid[0][0].accumulator == 3

    def test_multiple_pulses_accumulate(self):
        sa = SystolicArray(2)
        sa.pulse([1, 0], [2, 0])
        sa.pulse([1, 0], [2, 0])
        assert sa.grid[0][0].accumulator == 4


class TestSalu:
    def test_iadd4(self):
        salu = Salu(None)
        assert salu.execute(InstrSet.IAdd4, 10, 20) == 30

    def test_unknown_opcode_returns_zero(self):
        salu = Salu(None)
        assert salu.execute(InstrSet.FAdd32, 5, 5) == 0


class TestValu:
    def test_iadd4_vector(self):
        valu = Valu(None)
        result = valu.execute(InstrSet.IAdd4, [1, 2, 3, 4], [10, 20, 30, 40])
        assert result == [11, 22, 33, 44]

    def test_unknown_opcode_returns_zeros(self):
        valu = Valu(None)
        result = valu.execute(InstrSet.FAdd32, [1, 2, 3, 4], [1, 2, 3, 4])
        assert result == [0, 0, 0, 0]


# ===========================================================================
# 4. Control Flow Tests
# ===========================================================================

class TestProgramCounter:
    def test_initial_value(self):
        pc = ProgramCounter()
        assert pc.current_pc == 0

    def test_increment(self):
        pc = ProgramCounter()
        pc.increment()
        assert pc.current_pc == 1

    def test_multiple_increments(self):
        pc = ProgramCounter()
        for _ in range(5):
            pc.increment()
        assert pc.current_pc == 5

    def test_jump(self):
        pc = ProgramCounter()
        pc.jump(100)
        assert pc.current_pc == 100

    def test_jump_then_increment(self):
        pc = ProgramCounter()
        pc.jump(50)
        pc.increment()
        assert pc.current_pc == 51


class TestClock:
    def test_starts_at_zero(self):
        clk = Clock()
        assert clk.cycle == 0

    def test_tick_increments(self):
        clk = Clock()
        clk.tick()
        assert clk.cycle == 1

    def test_multiple_ticks(self):
        clk = Clock()
        for _ in range(10):
            clk.tick()
        assert clk.cycle == 10


class TestCommandSplitter:
    def test_execute_ops_go_to_execute_queue(self):
        cs = CommandSplitter()
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        cs.split([instr])
        assert len(cs.execute_queue) == 1

    def test_none_op_goes_to_access_queue(self):
        cs = CommandSplitter()
        instr = Instruction(Op.NONE, 0, 1, 2, TensorType.INT8)
        cs.split([instr])
        assert len(cs.access_queue) == 1

    def test_mixed_instructions_split_correctly(self):
        cs = CommandSplitter()
        instrs = [
            Instruction(Op.NONE, 0, 1, 2, TensorType.INT8),
            Instruction(Op.ADD, 0, 1, 3, TensorType.INT8),
            Instruction(Op.GEMM, 0, 1, 4, TensorType.INT8),
        ]
        cs.split(instrs)
        assert len(cs.access_queue) == 1
        assert len(cs.execute_queue) == 2


class TestCFGController:
    def test_handle_branch_returns_true_mask(self):
        cfg = CFGController()
        result = cfg.handle_branch(0b1100, 0b0011)
        assert result == 0b1100

    def test_handle_join_pops_false_mask(self):
        cfg = CFGController()
        cfg.handle_branch(0b1100, 0b0011)
        popped = cfg.handle_join()
        assert popped == 0b0011

    def test_nested_branch_join(self):
        cfg = CFGController()
        cfg.handle_branch(0xF0, 0x0F)
        cfg.handle_branch(0xC0, 0x30)
        assert cfg.handle_join() == 0x30
        assert cfg.handle_join() == 0x0F


# ===========================================================================
# 5. Scoreboard & Scheduler Tests
# ===========================================================================

class TestScoreboard:
    def setup_method(self):
        self.sb = Scoreboard()

    def test_initially_all_resources_ready(self):
        instr = Instruction(Op.ADD, 10, 20, 30, TensorType.INT8)
        assert self.sb.is_resource_ready(instr, 0) == True

    def test_dependency_blocks_issue(self):
        instr_prod = Instruction(Op.ADD, 0, 1, 5, TensorType.INT8)
        instr_cons = Instruction(Op.ADD, 5, 2, 6, TensorType.INT8)

        self.sb.mark_scheduled(instr_prod, 0, latency=4)
        # Consumer uses src1=5 which won't be ready until cycle 4
        assert self.sb.is_resource_ready(instr_cons, 2) == False

    def test_dependency_clears_after_latency(self):
        instr_prod = Instruction(Op.ADD, 0, 1, 5, TensorType.INT8)
        instr_cons = Instruction(Op.ADD, 5, 2, 6, TensorType.INT8)

        self.sb.mark_scheduled(instr_prod, 0, latency=4)
        assert self.sb.is_resource_ready(instr_cons, 4) == True

    def test_unit_conflict_blocks_issue(self):
        gemm1 = Instruction(Op.GEMM, 0, 1, 10, TensorType.INT8)
        gemm2 = Instruction(Op.GEMM, 2, 3, 11, TensorType.INT8)

        self.sb.mark_scheduled(gemm1, 0, latency=10)
        assert self.sb.is_resource_ready(gemm2, 5) == False

    def test_unit_free_after_latency(self):
        gemm1 = Instruction(Op.GEMM, 0, 1, 10, TensorType.INT8)
        gemm2 = Instruction(Op.GEMM, 2, 3, 11, TensorType.INT8)

        self.sb.mark_scheduled(gemm1, 0, latency=10)
        assert self.sb.is_resource_ready(gemm2, 10) == True


class TestBackScoreboardScheduler:
    def setup_method(self):
        self.sb = Scoreboard()
        self.scheduler = BackScoreboardScheduler(self.sb)

    def test_single_instr_dispatches_cycle_0(self):
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        self.scheduler.add_to_queue([instr])
        self.scheduler.schedule_cycle(0)
        assert self.scheduler.is_done()

    def test_dependent_instr_waits(self):
        prod = Instruction(Op.ADD, 0, 1, 5, TensorType.INT8)
        cons = Instruction(Op.ADD, 5, 2, 6, TensorType.INT8)

        self.scheduler.add_to_queue([prod, cons])
        self.scheduler.schedule_cycle(0)   # prod dispatches
        # cons cannot yet (src 5 not ready until cycle 0+2=2)
        assert not self.scheduler.is_done()

    def test_dependent_instr_dispatches_after_latency(self):
        prod = Instruction(Op.ADD, 0, 1, 5, TensorType.INT8)
        cons = Instruction(Op.ADD, 5, 2, 6, TensorType.INT8)

        self.scheduler.add_to_queue([prod, cons])
        self.scheduler.schedule_cycle(0)
        self.scheduler.schedule_cycle(2)   # prod latency=2
        assert self.scheduler.is_done()

    def test_independent_instrs_can_ooo_dispatch(self):
        """Two independent instructions should both be dispatchable."""
        instr_a = Instruction(Op.ADD, 0, 1, 10, TensorType.INT8)
        instr_b = Instruction(Op.ADD, 2, 3, 11, TensorType.INT8)

        self.scheduler.add_to_queue([instr_a, instr_b])
        # PLE can only run one ADD at a time; but they have no data deps
        self.scheduler.schedule_cycle(0)
        self.scheduler.schedule_cycle(2)
        assert self.scheduler.is_done()


# ===========================================================================
# 6. Semaphore & Sync Tests
# ===========================================================================

class TestSemaphore:
    def test_initial_value_zero(self):
        sem = Semaphore()
        assert sem.wait(0, 1) == False

    def test_signal_increments(self):
        sem = Semaphore()
        sem.signal(0)
        assert sem.wait(0, 1) == True

    def test_wait_multiple_signals(self):
        sem = Semaphore()
        sem.signal(0)
        sem.signal(0)
        assert sem.wait(0, 2) == True

    def test_wait_not_met(self):
        sem = Semaphore()
        sem.signal(0)
        assert sem.wait(0, 3) == False


class TestSyncManager:
    def setup_method(self):
        self.sm = SyncManager()

    def test_unit_not_blocked_initially(self):
        assert self.sm.is_unit_blocked("MCE") == False

    def test_unit_blocked_when_waiting(self):
        self.sm.register_wait("MCE", 0, 1)
        assert self.sm.is_unit_blocked("MCE") == True

    def test_unit_unblocked_after_signal(self):
        self.sm.register_wait("MCE", 0, 1)
        self.sm.signal(0)
        assert self.sm.is_unit_blocked("MCE") == False

    def test_multiple_units_independent(self):
        self.sm.register_wait("MCE", 0, 1)
        self.sm.register_wait("PLE", 1, 1)

        self.sm.signal(0)
        assert self.sm.is_unit_blocked("MCE") == False
        assert self.sm.is_unit_blocked("PLE") == True


# ===========================================================================
# 7. IR Graph & Memory Context Tests
# ===========================================================================

class TestTensor:
    def test_repr(self):
        t = Tensor(0, 64, 0, TensorType.INT8)
        assert "v0" in repr(t)
        assert "INT8" in repr(t)


class TestCtx:
    def test_allocate_assigns_incrementing_ids(self):
        ctx = Ctx(1024)
        t0 = ctx.allocate(64, TensorType.INT8)
        t1 = ctx.allocate(64, TensorType.INT8)
        assert t1.id == t0.id + 1

    def test_allocate_advances_offset(self):
        ctx = Ctx(1024)
        t0 = ctx.allocate(64, TensorType.INT8)
        t1 = ctx.allocate(32, TensorType.INT8)
        assert t1.offset == t0.offset + 64

    def test_tensor_type_preserved(self):
        ctx = Ctx(1024)
        t = ctx.allocate(16, TensorType.FLOAT32)
        assert t.type == TensorType.FLOAT32

    def test_tensor_size_preserved(self):
        ctx = Ctx(1024)
        t = ctx.allocate(128, TensorType.INT8)
        assert t.size == 128


# ===========================================================================
# 8. Compiler Tests
# ===========================================================================

class TestCompiler:
    def _make_ctx_with_add(self) -> Ctx:
        ctx = Ctx(1024)
        t0 = ctx.allocate(16, TensorType.INT8)
        t1 = ctx.allocate(16, TensorType.INT8)
        t2 = ctx.allocate(16, TensorType.INT8)
        node = Node(Op.ADD, [t0, t1], t2)
        ctx.nodes.append(node)
        return ctx

    def test_compile_produces_instructions(self):
        ctx = self._make_ctx_with_add()
        compiler = Compiler(ctx)
        instrs = compiler.compile()
        assert len(instrs) == 1

    def test_compiled_opcode_matches_node_op(self):
        ctx = self._make_ctx_with_add()
        compiler = Compiler(ctx)
        instrs = compiler.compile()
        assert instrs[0].opcode == Op.ADD

    def test_compiled_dest_matches_output_offset(self):
        ctx = Ctx(1024)
        t0 = ctx.allocate(16, TensorType.INT8)
        t1 = ctx.allocate(16, TensorType.INT8)
        t2 = ctx.allocate(16, TensorType.INT8)
        ctx.nodes.append(Node(Op.ADD, [t0, t1], t2))
        instrs = Compiler(ctx).compile()
        assert instrs[0].dest == t2.offset

    def test_compile_multiple_nodes(self):
        ctx = Ctx(1024)
        t0 = ctx.allocate(16, TensorType.INT8)
        t1 = ctx.allocate(16, TensorType.INT8)
        t2 = ctx.allocate(16, TensorType.INT8)
        t3 = ctx.allocate(16, TensorType.INT8)
        ctx.nodes.append(Node(Op.ADD, [t0, t1], t2))
        ctx.nodes.append(Node(Op.GEMM, [t2, t1], t3))
        instrs = Compiler(ctx).compile()
        assert len(instrs) == 2
        assert instrs[1].opcode == Op.GEMM

    def test_compile_single_input_uses_minus_one_src2(self):
        ctx = Ctx(1024)
        t0 = ctx.allocate(16, TensorType.INT8)
        t1 = ctx.allocate(16, TensorType.INT8)
        ctx.nodes.append(Node(Op.ADD, [t0], t1))   # only one input
        instrs = Compiler(ctx).compile()
        assert instrs[0].src2 == -1


# ===========================================================================
# 9. Dispatcher Cost Tests
# ===========================================================================

class TestDispatcher:
    def setup_method(self):
        self.sram = ScratchPad(1024)
        self.dispatcher = Dispatcher(self.sram)

    def test_gemm_int8_cost(self):
        instr = Instruction(Op.GEMM, 0, 1, 2, TensorType.INT8)
        cost = self.dispatcher.calculate_cost(instr, 1024)
        assert cost > 0

    def test_gemm_float32_costs_more_than_int8(self):
        instr_int8 = Instruction(Op.GEMM, 0, 1, 2, TensorType.INT8)
        instr_fp32 = Instruction(Op.GEMM, 0, 1, 2, TensorType.FLOAT32)
        cost_int8 = self.dispatcher.calculate_cost(instr_int8, 1024)
        cost_fp32 = self.dispatcher.calculate_cost(instr_fp32, 1024)
        assert cost_fp32 > cost_int8

    def test_add_op_cost(self):
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        cost = self.dispatcher.calculate_cost(instr, 256)
        assert cost > 0

    def test_dispatch_block_accumulates_cycles(self):
        block = Block(0)
        block.add_instr(Instruction(Op.ADD, 0, 1, 2, TensorType.INT8))
        block.add_instr(Instruction(Op.GEMM, 0, 1, 3, TensorType.INT8))
        self.dispatcher.dispatch_block(block)
        assert self.dispatcher.total_cycles > 0


# ===========================================================================
# 10. NpuVirtualDevice Tests
# ===========================================================================

class TestNpuVirtualDevice:
    def setup_method(self):
        self.config = HardwareConfig()
        self.device = NpuVirtualDevice(self.config)

    def test_initial_cycle_zero(self):
        assert self.device.cycle == 0

    def test_tick_increments_cycle(self):
        self.device.tick(deque())
        assert self.device.cycle == 1

    def test_tick_fetches_from_command_queue(self):
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        q = deque([instr])
        self.device.tick(q)
        assert len(q) == 0  # Consumed

    def test_calculate_latency_gemm(self):
        instr = Instruction(Op.GEMM, 0, 1, 2, TensorType.INT8)
        lat = self.device.calculate_latency(instr)
        assert lat >= 4

    def test_calculate_latency_add_is_one(self):
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        lat = self.device.calculate_latency(instr)
        assert lat == 1

    def test_max_inflight_respected(self):
        """Fill queue beyond MAX_INFLIGHT_INSTR: device should only consume up to limit."""
        q = deque([
            Instruction(Op.ADD, i, i+1, i+2, TensorType.INT8)
            for i in range(self.config.MAX_INFLIGHT_INSTR + 10)
        ])
        total = len(q)
        self.device.tick(q)
        # Only 1 instruction fetched per tick in simple model
        assert len(q) == total - 1


# ===========================================================================
# 11. Block & Instruction Tests
# ===========================================================================

class TestInstruction:
    def test_repr(self):
        instr = Instruction(Op.ADD, 1, 2, 3, TensorType.INT8)
        r = repr(instr)
        assert "ADD" in r
        assert "S1:1" in r
        assert "D:3" in r


class TestBlock:
    def test_add_instr(self):
        block = Block(0)
        instr = Instruction(Op.ADD, 0, 1, 2, TensorType.INT8)
        block.add_instr(instr)
        assert len(block.instructions) == 1

    def test_block_id(self):
        block = Block(42)
        assert block.block_id == 42

    def test_next_block_default_none(self):
        block = Block(0)
        assert block.next_block is None

    def test_link_blocks(self):
        b0 = Block(0)
        b1 = Block(1)
        b0.next_block = b1
        assert b0.next_block is b1


# ===========================================================================
# 12. HardwareConfig Tests
# ===========================================================================

class TestHardwareConfig:
    def test_sram_total_size(self):
        config = HardwareConfig()
        assert config.SRAM_TOTAL_SIZE == config.SRAM_BANKS * config.BANK_SIZE

    def test_default_mce_units(self):
        config = HardwareConfig()
        assert config.MCE_UNITS == 256

    def test_pipeline_stages(self):
        config = HardwareConfig()
        assert config.PIPELINE_STAGES == 5

""""
# ===========================================================================
# Run
# ===========================================================================
def execute():

    SEP = "=" * 56

    # =========================================================
    # Demo 1 — Simple GEMM + ADD Graph End-to-End
    # =========================================================
    print(f"\n{SEP}")
    print(" Demo 1 · GEMM + ADD Graph  (Ctx → Compiler → Instrs)")
    print(SEP)

    ctx = Ctx(alloc_size=4096)

    # Allocate tensors  (weights · activations · output · bias · result)
    W  = ctx.allocate(256, TensorType.INT8)   # weight matrix  [16×16]
    X  = ctx.allocate(256, TensorType.INT8)   # input  matrix  [16×16]
    Y  = ctx.allocate(256, TensorType.INT8)   # GEMM   output
    B  = ctx.allocate(16,  TensorType.INT8)   # bias   vector
    Z  = ctx.allocate(256, TensorType.INT8)   # ADD    output

    # Build IR nodes manually (GraphBuilder __ini__ typo kept intact)
    ctx.nodes.append(Node(Op.GEMM, [W, X], Y))   # Y  = W @ X
    ctx.nodes.append(Node(Op.ADD,  [Y, B], Z))   # Z  = Y + B

    print(f"  IR nodes  : {len(ctx.nodes)}")
    for n in ctx.nodes:
        ins = ", ".join(repr(t) for t in n.inputs)
        print(f"    {n.op.name:<6}  inputs=[{ins}]  →  {n.output!r}")

    # Compile → ISA instructions
    compiler = Compiler(ctx)
    instrs   = compiler.compile()

    print(f"\n  Compiled  : {len(instrs)} instruction(s)")
    for i, ins in enumerate(instrs):
        print(f"    [{i}] {ins}")

    # =========================================================
    # Demo 2 — Scoreboard Scheduling with Dependency Stalls
    # =========================================================
    print(f"\n{SEP}")
    print(" Demo 2 · Scoreboard Scheduling  (RAW hazard stall)")
    print(SEP)

    sb        = Scoreboard()
    scheduler = BackScoreboardScheduler(sb)

    #  instr_a : ADD   src(0,1) → dest 5      (produces reg 5)
    #  instr_b : ADD   src(5,2) → dest 6      (RAW dep on reg 5)
    #  instr_c : GEMM  src(3,4) → dest 7      (independent)
    instr_a = Instruction(Op.ADD,  0, 1, 5, TensorType.INT8)
    instr_b = Instruction(Op.ADD,  5, 2, 6, TensorType.INT8)
    instr_c = Instruction(Op.GEMM, 3, 4, 7, TensorType.INT8)

    scheduler.add_to_queue([instr_a, instr_b, instr_c])

    cycle = 0
    print(f"  {'Cycle':<6} {'Queue':>5}   dispatched")
    while not scheduler.is_done():
        before = len(scheduler.issue_queue)
        scheduler.schedule_cycle(cycle)
        after  = len(scheduler.issue_queue)
        fired  = before - after
        tag    = f"{fired} instr(s)" if fired else "-- stall --"
        print(f"  {cycle:<6} {after:>5}   {tag}")
        cycle += 1
        if cycle > 30:      # safety guard
            break

    print(f"\n  Total cycles to drain queue : {cycle}")
    for instr, start, lat in scheduler.executed_instructions:
        print(f"    {instr.opcode.name:<6} dest={instr.dest}  "
              f"issued@{start}  latency={lat}  done@{start+lat}")

    # =========================================================
    # Demo 3 — DMA → SRAM → Execute Pipeline
    # =========================================================
    print(f"\n{SEP}")
    print(" Demo 3 · DMA → SRAM → Execute  (weight + activation load)")
    print(SEP)

    hw      = HardwareConfig()
    sram    = ScratchPad(hw.SRAM_TOTAL_SIZE)
    dma     = DMA(sram)
    core    = NPUCore(memory_size=hw.SRAM_TOTAL_SIZE)

    # Populate DRAM with synthetic weights (0..15) and activations (1..16)
    WEIGHT_DRAM  = 0x0000
    ACT_DRAM     = 0x1000
    WEIGHT_SRAM  = 0
    ACT_SRAM     = 16

    for i in range(16):
        dma.dram[WEIGHT_DRAM + i] = float(i)
        dma.dram[ACT_DRAM    + i] = float(i + 1)

    w_cycles = dma.copy_to_sram(WEIGHT_DRAM, WEIGHT_SRAM, 16, is_weight=True)
    a_cycles = dma.copy_to_sram(ACT_DRAM,    ACT_SRAM,    16, is_weight=False)
    total_dma = w_cycles + a_cycles

    print(f"  DMA weight transfer : {w_cycles} cycles")
    print(f"  DMA activation xfer : {a_cycles} cycles")
    print(f"  Total DMA cost      : {total_dma} cycles")

    # Fire an ADD instruction over the loaded data
    DEST_SRAM = 32
    add_instr = Instruction(Op.ADD, WEIGHT_SRAM, ACT_SRAM, DEST_SRAM, TensorType.INT8)
    core.sram = sram        # point core at the shared scratchpad
    core.step(add_instr)

    print(f"\n  ADD executed @ core PC={core.pc.current_pc}  clock={core.clock.cycle}")
    print(f"  Output[0..4] : ", end="")
    print([sram.read(DEST_SRAM + i) for i in range(4)])   # expect [1,3,5,7]

    # =========================================================
    # Demo 4 — Cycle-Accurate NpuVirtualDevice Simulation
    # =========================================================
    print(f"\n{SEP}")
    print(" Demo 4 · Cycle-Accurate NpuVirtualDevice")
    print(SEP)

    device = NpuVirtualDevice(HardwareConfig())
    # Mix of ADD (PLE, latency=1) and GEMM (MCE, latency≥4)
    cmd_q = deque([
        Instruction(Op.ADD,  0,  1,  10, TensorType.INT8),
        Instruction(Op.GEMM, 10, 2,  11, TensorType.INT8),
        Instruction(Op.ADD,  11, 3,  12, TensorType.INT8),
        Instruction(Op.GEMM, 4,  5,  13, TensorType.INT8),
        Instruction(Op.ADD,  13, 6,  14, TensorType.INT8),
    ])
    total_instrs = len(cmd_q)

    print(f"  Simulating {total_instrs} instructions …\n")
    print(f"  {'Cycle':<6} {'In-flight':>9} {'Retired':>8} {'Remaining':>10}")

    MAX_CYCLES = 64
    for _ in range(MAX_CYCLES):
        device.tick(cmd_q)
        remaining = len(cmd_q)
        print(f"  {device.cycle:<6} {len(device.in_flight):>9} "
              f"{device.retired_count:>8} {remaining:>10}")
        if remaining == 0 and len(device.in_flight) == 0:
            break

    print(f"\n  Finished at cycle {device.cycle}  "
          f"· retired {device.retired_count} instruction(s)")

"""
# ===========================================================================
# EventLogger — stdout event tracing for NPU simulator verification
# ===========================================================================

from enum import Enum, auto
from typing import Optional

class EventKind(Enum):
    # Pipeline
    STAGE_TRANSITION  = auto()   # instr moved between pipeline stages
    # Scoreboard
    HAZARD_STALL      = auto()   # instr blocked due to RAW / unit conflict
    HAZARD_CLEAR      = auto()   # stall resolved, instr is now dispatchable
    # DMA
    DMA_START         = auto()   # transfer kicked off
    DMA_END           = auto()   # transfer completed
    # Instruction lifecycle
    INSTR_DISPATCH    = auto()   # instr leaves issue queue → execution unit
    INSTR_EXECUTE     = auto()   # instr enters execute stage
    INSTR_RETIRE      = auto()   # instr writeback complete


# ANSI colour codes — one colour per event category
_COLOUR = {
    EventKind.STAGE_TRANSITION : "\033[36m",   # cyan
    EventKind.HAZARD_STALL     : "\033[33m",   # yellow
    EventKind.HAZARD_CLEAR     : "\033[32m",   # green
    EventKind.DMA_START        : "\033[34m",   # blue
    EventKind.DMA_END          : "\033[34m",   # blue
    EventKind.INSTR_DISPATCH   : "\033[35m",   # magenta
    EventKind.INSTR_EXECUTE    : "\033[35m",   # magenta
    EventKind.INSTR_RETIRE     : "\033[32m",   # green
}
_RESET = "\033[0m"


class EventLogger:
    """
    Singleton-style logger.  Pass one instance into every hardware unit
    that needs to emit events.  All output goes to stdout immediately.

    Usage
    -----
        logger = EventLogger(use_colour=True)
        dma    = InstrumentedDMA(sram, logger)
        sb     = InstrumentedScoreboard(logger)
        ...
    """

    # Column widths for aligned output
    _W_CYCLE = 6
    _W_KIND  = 20
    _W_UNIT  = 8

    def __init__(self, use_colour: bool = True, min_kind: Optional[EventKind] = None):
        self.use_colour  = use_colour
        self.min_kind    = min_kind   # future: severity filter
        self._last_cycle = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self,
            cycle: int,
            kind: EventKind,
            unit: str,
            msg: str,
            instr: Optional["Instruction"] = None):
        """
        Emit a single event line immediately to stdout.

        Format:
            [  42] INSTR_DISPATCH   MCE      [GEMM | S1:0 S2:1 -> D:10]  msg
        """
        # Separator when cycle advances — makes it easy to scan per-cycle
        if cycle != self._last_cycle:
            if self._last_cycle != -1:
                print(f"{'':>{self._W_CYCLE + 2}}{'·' * 48}")
            self._last_cycle = cycle

        colour = _COLOUR.get(kind, "") if self.use_colour else ""
        reset  = _RESET if self.use_colour else ""

        cycle_col = f"[{cycle:>{self._W_CYCLE - 2}}]"
        kind_col  = f"{kind.name:<{self._W_KIND}}"
        unit_col  = f"{unit:<{self._W_UNIT}}"
        instr_col = f"  {instr!r}" if instr else ""

        print(f"{colour}{cycle_col} {kind_col} {unit_col}{instr_col}  {msg}{reset}")

    def separator(self, label: str = ""):
        w = 56
        bar = "═" * w
        if label:
            pad = (w - len(label) - 2) // 2
            print(f"\n╔{'═' * pad} {label} {'═' * pad}╗")
        else:
            print(f"\n{bar}")


# ===========================================================================
# Instrumented hardware units
# Each wraps the original class and calls logger.log() at key points.
# The originals are untouched — drop-in replacement in execute().
# ===========================================================================

class InstrumentedDMA(DMA):
    """DMA with DMA_START / DMA_END events."""

    def __init__(self, sram: ScratchPad, logger: EventLogger):
        super().__init__(sram)
        self._log = logger
        self._cycle = 0   # caller must update .cycle each tick

    def copy_to_sram(self, dram_addr: int, sram_addr: int,
                     size: int, is_weight: bool) -> int:
        kind = "weight" if is_weight else "activation"
        self._log.log(self._cycle, EventKind.DMA_START, "DMA",
                      f"DRAM[{dram_addr:#06x}] → SRAM[{sram_addr:#06x}]  "
                      f"{size}B  ({kind})")

        cost = super().copy_to_sram(dram_addr, sram_addr, size, is_weight)

        self._log.log(self._cycle + cost, EventKind.DMA_END, "DMA",
                      f"done  cost={cost} cycles  {size}B  ({kind})")
        return cost

    def copy_to_dram(self, sram_addr: int, dram_addr: int,
                     size: int, is_weight: bool) -> int:
        self._log.log(self._cycle, EventKind.DMA_START, "DMA",
                      f"SRAM[{sram_addr:#06x}] → DRAM[{dram_addr:#06x}]  {size}B")

        cost = super().copy_to_dram(sram_addr, dram_addr, size, is_weight)

        self._log.log(self._cycle + cost, EventKind.DMA_END, "DMA",
                      f"done  cost={cost} cycles  {size}B")
        return cost


class InstrumentedScoreboard(Scoreboard):
    """Scoreboard that emits HAZARD_STALL and HAZARD_CLEAR events."""

    def __init__(self, logger: EventLogger):
        super().__init__()
        self._log   = logger
        self._cycle = 0   # caller must update before calling is_resource_ready

    def is_resource_ready(self, instr: "Instruction", current_cycle: int) -> bool:
        self._cycle = current_cycle
        ready = super().is_resource_ready(instr, current_cycle)

        if not ready:
            # Diagnose which dependency is blocking
            src1_rdy = self.register_ready_cycle.get(instr.src1, 0) <= current_cycle
            src2_rdy = self.register_ready_cycle.get(instr.src2, 0) <= current_cycle
            unit     = self._get_unit_for_op(instr.opcode)
            unit_rdy = self.unit_busy_until[unit] <= current_cycle

            reasons = []
            if not src1_rdy:
                rdy_at = self.register_ready_cycle[instr.src1]
                reasons.append(f"src1=r{instr.src1} ready@{rdy_at}")
            if not src2_rdy:
                rdy_at = self.register_ready_cycle[instr.src2]
                reasons.append(f"src2=r{instr.src2} ready@{rdy_at}")
            if not unit_rdy:
                free_at = self.unit_busy_until[unit]
                reasons.append(f"{unit} busy until {free_at}")

            self._log.log(current_cycle, EventKind.HAZARD_STALL,
                          unit, "  stall: " + ", ".join(reasons), instr)
        else:
            # Only emit CLEAR if this instr had previously been seen stalling
            unit = self._get_unit_for_op(instr.opcode)
            self._log.log(current_cycle, EventKind.HAZARD_CLEAR,
                          unit, "  ready to dispatch", instr)

        return ready


class InstrumentedScheduler(BackScoreboardScheduler):
    """Scheduler that emits INSTR_DISPATCH events."""

    def __init__(self, scoreboard: InstrumentedScoreboard, logger: EventLogger):
        super().__init__(scoreboard)
        self._log = logger

    def schedule_cycle(self, current_cycle: int):
        # Patch scoreboard cycle so stall messages carry the right cycle number
        # self.scoreboard._cycle = current_cycle

        before = len(self.issue_queue)
        super().schedule_cycle(current_cycle)
        after  = len(self.issue_queue)

        # Emit a DISPATCH event for each instruction that was just issued
        newly_dispatched = self.executed_instructions[-(before - after):]
        for instr, start, latency in newly_dispatched:
            unit = self.scoreboard._get_unit_for_op(instr.opcode)
            self._log.log(current_cycle, EventKind.INSTR_DISPATCH, unit,
                          f"latency={latency}  done@{start + latency}", instr)


class InstrumentedNpuVirtualDevice(NpuVirtualDevice):
    """
    NpuVirtualDevice that emits STAGE_TRANSITION, INSTR_EXECUTE,
    and INSTR_RETIRE events on every tick.
    """

    def __init__(self, config: HardwareConfig, logger: EventLogger):
        super().__init__(config)
        self._log = logger
        # Track previous stage per in-flight instr id to detect transitions
        self._prev_state: dict = {}

    def tick(self, command_queue: deque):
        # Snapshot states before the tick so we can detect transitions after
        pre = {id(op): op.state for op in self.in_flight}

        super().tick(command_queue)

        post = {id(op): op.state for op in self.in_flight}

        # --- Retirement: ops that disappeared from in_flight this tick ---
        for op_id, old_state in pre.items():
            if op_id not in post:
                # find the instruction from pre-snapshot via id (it's gone now)
                # We log it from the executed_instructions if available, or skip
                self._log.log(self.cycle, EventKind.INSTR_RETIRE, "ROB",
                              f"retired  (was {old_state.name})")

        # --- Stage transitions for remaining in-flight instructions ---
        for op in self.in_flight:
            oid      = id(op)
            new_state = op.state
            old_state = pre.get(oid)          # None if just fetched this tick

            if old_state is None:
                # Freshly fetched
                self._log.log(self.cycle, EventKind.STAGE_TRANSITION, "FETCH",
                              f"→ {new_state.name}", op.instr)

            elif old_state != new_state:
                unit = self.scoreboard._get_unit_for_op(op.instr.opcode) \
                       if hasattr(self.scoreboard, '_get_unit_for_op') else "PIPE"

                kind = EventKind.INSTR_EXECUTE \
                       if new_state == PipelineState.EXECUTE else \
                       EventKind.STAGE_TRANSITION

                self._log.log(self.cycle, kind, unit,
                              f"{old_state.name} → {new_state.name}"
                              f"  rem={op.cycles_remaining}", op.instr)


# ===========================================================================
# execute() — all four demos with full event logging
# ===========================================================================

def execute():

    logger = EventLogger(use_colour=True)
    SEP    = "=" * 56

    # =========================================================
    # Demo 1 — GEMM + ADD Graph  (Ctx → Compiler → Instrs)
    # =========================================================
    logger.separator("Demo 1 · GEMM + ADD Graph")

    ctx = Ctx(alloc_size=4096)
    W   = ctx.allocate(256, TensorType.INT8)
    X   = ctx.allocate(256, TensorType.INT8)
    Y   = ctx.allocate(256, TensorType.INT8)
    B   = ctx.allocate(16,  TensorType.INT8)
    Z   = ctx.allocate(256, TensorType.INT8)

    ctx.nodes.append(Node(Op.GEMM, [W, X], Y))
    ctx.nodes.append(Node(Op.ADD,  [Y, B], Z))

    compiler = Compiler(ctx)
    instrs   = compiler.compile()

    # Log each compiled instruction as a DISPATCH event at cycle 0
    for ins in instrs:
        unit = "MCE" if ins.opcode == Op.GEMM else "PLE"
        logger.log(0, EventKind.INSTR_DISPATCH, unit,
                   f"compiled → {ins.opcode.name}", ins)

    # =========================================================
    # Demo 2 — Scoreboard Scheduling with Dependency Stalls
    # =========================================================
    logger.separator("Demo 2 · Scoreboard Hazard Stalls")

    sb        = InstrumentedScoreboard(logger)
    scheduler = InstrumentedScheduler(sb, logger)

    instr_a = Instruction(Op.ADD,  0, 1, 5, TensorType.INT8)
    instr_b = Instruction(Op.ADD,  5, 2, 6, TensorType.INT8)
    instr_c = Instruction(Op.GEMM, 3, 4, 7, TensorType.INT8)

    scheduler.add_to_queue([instr_a, instr_b, instr_c])

    cycle = 0
    while not scheduler.is_done():
        scheduler.schedule_cycle(cycle)
        cycle += 1
        if cycle > 30:
            break

    # =========================================================
    # Demo 3 — DMA → SRAM → Execute
    # =========================================================
    logger.separator("Demo 3 · DMA → SRAM → Execute")

    hw   = HardwareConfig()
    sram = ScratchPad(hw.SRAM_TOTAL_SIZE)
    dma  = InstrumentedDMA(sram, logger)
    core = NPUCore(memory_size=hw.SRAM_TOTAL_SIZE)

    WEIGHT_DRAM, ACT_DRAM   = 0x0000, 0x1000
    WEIGHT_SRAM, ACT_SRAM   = 0, 16

    for i in range(16):
        dma.dram[WEIGHT_DRAM + i] = float(i)
        dma.dram[ACT_DRAM    + i] = float(i + 1)

    dma._cycle = 0
    w_cost = dma.copy_to_sram(WEIGHT_DRAM, WEIGHT_SRAM, 16, is_weight=True)
    dma._cycle = w_cost
    dma.copy_to_sram(ACT_DRAM, ACT_SRAM, 16, is_weight=False)

    DEST_SRAM = 32
    add_instr = Instruction(Op.ADD, WEIGHT_SRAM, ACT_SRAM, DEST_SRAM, TensorType.INT8)
    core.sram = sram
    core.step(add_instr)

    logger.log(w_cost, EventKind.INSTR_EXECUTE, "PLE",
               f"ADD executed  output[0..3]={[sram.read(DEST_SRAM+i) for i in range(4)]}",
               add_instr)

    # =========================================================
    # Demo 4 — Cycle-Accurate NpuVirtualDevice
    # =========================================================
    logger.separator("Demo 4 · Cycle-Accurate Pipeline")

    device = InstrumentedNpuVirtualDevice(HardwareConfig(), logger)

    cmd_q = deque([
        Instruction(Op.ADD,  0,  1,  10, TensorType.INT8),
        Instruction(Op.GEMM, 10, 2,  11, TensorType.INT8),
        Instruction(Op.ADD,  11, 3,  12, TensorType.INT8),
        Instruction(Op.GEMM, 4,  5,  13, TensorType.INT8),
        Instruction(Op.ADD,  13, 6,  14, TensorType.INT8),
    ])

    for _ in range(64):
        device.tick(cmd_q)
        if not cmd_q and not device.in_flight:
            break

    logger.separator(f"done — retired {device.retired_count} instrs @ cycle {device.cycle}")


if __name__ == "__main__":
    execute()
