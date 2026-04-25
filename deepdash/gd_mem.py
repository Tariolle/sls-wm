"""Read Geometry Dash game state from process memory (Windows, GD 2.2074 x64).

Pointer chain (extracted from GeometryDash.exe disassembly + geode-sdk/bindings):
    [GD.exe + GM_SINGLETON]  -> GameManager*
    [GameManager + 0x208]    -> PlayLayer*       (m_playLayer)
    [PlayLayer  + 0xD98]     -> PlayerObject*    (m_player1)
    [PlayerObject + 0x9C0]   -> bool             (m_isDead)
"""

import ctypes
import ctypes.wintypes as wt
import struct

# --- Offsets for GD 2.2074 x64 ---
# To update for a different version, re-extract from the binary using
# scripts/calibrate_gd_offsets.py or Cheat Engine.
GM_SINGLETON_RVA = 0x6A4E68  # RVA of the GameManager* global
OFF_PLAY_LAYER = 0x208       # GameManager -> m_playLayer
OFF_PLAYER1 = 0xD98          # GJBaseGameLayer -> m_player1
OFF_IS_DEAD = 0x9C0          # PlayerObject -> m_isDead

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
SNAP_MODULE = 0x08
SNAP_MODULE_32 = 0x10

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Set argtypes for ReadProcessMemory to handle 64-bit addresses
kernel32.ReadProcessMemory.argtypes = [
    wt.HANDLE, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.ReadProcessMemory.restype = wt.BOOL


class ModuleEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("th32ModuleID", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("GlblcntUsage", wt.DWORD),
        ("ProccntUsage", wt.DWORD),
        ("modBaseAddr", ctypes.POINTER(wt.BYTE)),
        ("modBaseSize", wt.DWORD),
        ("hModule", wt.HMODULE),
        ("szModule", ctypes.c_char * 256),
        ("szExePath", ctypes.c_char * wt.MAX_PATH),
    ]


class ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(wt.ULONG)),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", wt.LONG),
        ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_char * wt.MAX_PATH),
    ]


def _find_process(name: str) -> int:
    """Return PID of a process by executable name."""
    snap = kernel32.CreateToolhelp32Snapshot(0x02, 0)
    entry = ProcessEntry()
    entry.dwSize = ctypes.sizeof(entry)
    if not kernel32.Process32First(snap, ctypes.byref(entry)):
        kernel32.CloseHandle(snap)
        raise RuntimeError(f"Process32First failed")
    while True:
        if entry.szExeFile.decode("utf-8", "replace").lower() == name.lower():
            pid = entry.th32ProcessID
            kernel32.CloseHandle(snap)
            return pid
        if not kernel32.Process32Next(snap, ctypes.byref(entry)):
            break
    kernel32.CloseHandle(snap)
    raise RuntimeError(f"Process '{name}' not found")


def _get_base_address(pid: int, module_name: str) -> int:
    """Return the base address of a module in the given process."""
    snap = kernel32.CreateToolhelp32Snapshot(SNAP_MODULE | SNAP_MODULE_32, pid)
    entry = ModuleEntry()
    entry.dwSize = ctypes.sizeof(entry)
    if not kernel32.Module32First(snap, ctypes.byref(entry)):
        kernel32.CloseHandle(snap)
        raise RuntimeError("Module32First failed")
    while True:
        if entry.szModule.decode("utf-8", "replace").lower() == module_name.lower():
            addr = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value
            kernel32.CloseHandle(snap)
            return addr
        if not kernel32.Module32Next(snap, ctypes.byref(entry)):
            break
    kernel32.CloseHandle(snap)
    raise RuntimeError(f"Module '{module_name}' not found in PID {pid}")


def _read_u64(handle: int, addr: int) -> int:
    buf = ctypes.create_string_buffer(8)
    n = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(handle, addr, buf, 8, ctypes.byref(n))
    if not ok:
        return 0
    return struct.unpack("<Q", buf.raw)[0]


def _read_u8(handle: int, addr: int):
    """Read one byte from the target process. Returns None on failure so
    callers can distinguish a real zero (alive) from a failed read."""
    buf = ctypes.create_string_buffer(1)
    n = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(handle, addr, buf, 1, ctypes.byref(n))
    if not ok:
        return None
    return buf.raw[0]


class GDReader:
    """Read game state from a running Geometry Dash process."""

    def __init__(self, process_name: str = "GeometryDash.exe"):
        self.pid = _find_process(process_name)
        self.handle = kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, self.pid
        )
        if not self.handle:
            raise RuntimeError(f"OpenProcess failed for PID {self.pid}")
        self.base = _get_base_address(self.pid, process_name)
        self.gm_ptr_addr = self.base + GM_SINGLETON_RVA
        print(f"GDReader: PID={self.pid}, base=0x{self.base:X}, "
              f"GM@0x{self.gm_ptr_addr:X}")

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def _follow_chain(self, *offsets) -> int:
        """Dereference a pointer chain: base_ptr -> [+off1] -> [+off2] -> ..."""
        addr = self.gm_ptr_addr
        for i, off in enumerate(offsets):
            ptr = _read_u64(self.handle, addr)
            if ptr == 0:
                return 0
            addr = ptr + off
        return addr

    def is_in_level(self) -> bool:
        """True if a PlayLayer exists (player is in a level)."""
        gm = _read_u64(self.handle, self.gm_ptr_addr)
        if gm == 0:
            return False
        play_layer = _read_u64(self.handle, gm + OFF_PLAY_LAYER)
        return play_layer != 0

    def is_dead(self) -> bool:
        """True if m_player1->m_isDead is set. Treats RPM failures as
        terminal (not alive) to avoid hiding deploy bugs."""
        addr = self._follow_chain(OFF_PLAY_LAYER, OFF_PLAYER1, OFF_IS_DEAD)
        if addr == 0:
            return False
        b = _read_u8(self.handle, addr)
        return b is None or b != 0

    def get_state(self) -> dict:
        """Read full game state: in_level, is_dead."""
        gm = _read_u64(self.handle, self.gm_ptr_addr)
        if gm == 0:
            return {"in_level": False, "is_dead": False}
        play_layer = _read_u64(self.handle, gm + OFF_PLAY_LAYER)
        if play_layer == 0:
            return {"in_level": False, "is_dead": False}
        player1 = _read_u64(self.handle, play_layer + OFF_PLAYER1)
        if player1 == 0:
            return {"in_level": True, "is_dead": False}
        b = _read_u8(self.handle, player1 + OFF_IS_DEAD)
        is_dead = b is None or b != 0
        return {"in_level": True, "is_dead": is_dead}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
