/** Keyboard + pointer-lock mouse state. Edge-triggered presses are cleared each frame by endFrame(). */
export class Input {
  private down = new Set<string>();
  private pressed = new Set<string>();
  mouseDX = 0;
  mouseDY = 0;
  fire = false;
  firePressed = false;
  aim = false;
  locked = false;
  /** Set when pointer lock is unavailable (e.g. sandboxed iframe): raw mouse deltas are used instead. */
  freeLook = false;

  constructor(private el: HTMLElement) {
    window.addEventListener('keydown', (e) => {
      if (!this.down.has(e.code)) this.pressed.add(e.code);
      this.down.add(e.code);
      if (this.active && (e.code === 'Space' || e.code.startsWith('Arrow'))) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.down.delete(e.code));
    window.addEventListener('blur', () => {
      this.down.clear();
      this.fire = false;
      this.aim = false;
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.active) return;
      this.mouseDX += e.movementX;
      this.mouseDY += e.movementY;
    });
    el.addEventListener('mousedown', (e) => {
      if (!this.active) return;
      if (e.button === 0) {
        this.fire = true;
        this.firePressed = true;
      }
      if (e.button === 2) this.aim = true;
    });
    window.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.fire = false;
      if (e.button === 2) this.aim = false;
    });
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === this.el;
      if (this.locked) this.freeLook = false;
      if (!this.locked) {
        this.fire = false;
        this.aim = false;
      }
    });
  }

  get active() {
    return this.locked || this.freeLook;
  }

  requestLock() {
    const fallback = () => {
      if (!this.locked) this.freeLook = true;
    };
    try {
      const p = this.el.requestPointerLock?.() as unknown as Promise<void> | undefined;
      p?.catch?.(fallback);
    } catch {
      fallback();
    }
    setTimeout(fallback, 400);
  }

  key(code: string) {
    return this.down.has(code);
  }

  keyPressed(code: string) {
    return this.pressed.has(code);
  }

  endFrame() {
    this.pressed.clear();
    this.firePressed = false;
    this.mouseDX = 0;
    this.mouseDY = 0;
  }
}
