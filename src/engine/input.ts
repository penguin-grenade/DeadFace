/**
 * Keyboard + pointer-lock mouse state, plus what the on-screen touch controls (touch.ts) feed in.
 * Edge-triggered presses are cleared each frame by endFrame().
 */
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
  /** Playing with the touch controls: no pointer lock, and the mouse handlers stand down. */
  touch = false;
  /** Touch stick: x right, y forward, within the unit circle (0 when unused). */
  moveX = 0;
  moveY = 0;
  sprint = false;
  crouch = false;

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
      this.moveX = this.moveY = 0;
      this.sprint = false;
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.active || this.touch) return;
      this.mouseDX += e.movementX;
      this.mouseDY += e.movementY;
    });
    el.addEventListener('mousedown', (e) => {
      if (!this.active || this.touch) return;
      if (e.button === 0) {
        this.fire = true;
        this.firePressed = true;
      }
      if (e.button === 2) this.aim = true;
    });
    window.addEventListener('mouseup', (e) => {
      if (this.touch) return;
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
    return this.locked || this.freeLook || this.touch;
  }

  /** A key press from an on-screen button: keyPressed() sees it for one frame. */
  press(code: string) {
    this.pressed.add(code);
  }

  /** Look, in mouse pixels (a touch drag). */
  look(dx: number, dy: number) {
    this.mouseDX += dx;
    this.mouseDY += dy;
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
