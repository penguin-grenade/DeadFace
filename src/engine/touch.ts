import type { Input } from './input';

/**
 * On-screen controls for phones and tablets, feeding the same Input the keyboard and mouse do.
 *
 * Left thumb: a floating stick that starts wherever the thumb lands on the left of the screen
 * (analog, so a small push walks slowly); pushing past its ring towards the top sprints.
 * Right thumb: drag anywhere to look. Buttons round the bottom-right corner fire, aim (toggle),
 * reload, jump, crouch (toggle) and switch the weapon light; dragging the fire button also looks,
 * so you can adjust aim while shooting. A second fire button sits above the stick for firing
 * with the left thumb while the right one aims, and the top-left button pauses.
 */

type Act = 'fire' | 'aim' | 'reload' | 'jump' | 'crouch' | 'light' | 'pause';

type Role =
  | { kind: 'move' }
  | { kind: 'look'; x: number; y: number }
  | { kind: 'fire'; x: number; y: number; btn: HTMLElement }
  | { kind: 'button'; btn: HTMLElement; act: Act };

/** What the buttons show; filled in by the game every frame. */
export interface TouchHud {
  ammo: number;
  reserve: number;
  reloading: boolean;
  /** 0..1, how far the weapon is raised: look slows down while aiming. */
  aim: number;
  light: boolean;
}

// 24 px line icons, stroked in the text colour.
const ICON: Record<Act, string> = {
  fire: '<path d="M9.5 20.5h5V11L12 4l-2.5 7z"/><path d="M9.5 16.5h5"/>',
  aim: '<circle cx="12" cy="12" r="6"/><path d="M12 2.5v5M12 16.5v5M2.5 12h5M16.5 12h5"/><circle cx="12" cy="12" r=".6"/>',
  reload: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.8 4.2v4.6h-4.6"/>',
  jump: '<path d="M6 15l6-6 6 6"/><path d="M6 20l6-6 6 6" opacity=".45"/>',
  crouch: '<path d="M6 8l6 6 6-6"/><path d="M5 19.5h14"/>',
  light: '<path d="M3.5 9.5h7l3.5-3v11l-3.5-3h-7z"/><path d="M17 8.5l3-1.8M17.5 12h3.5M17 15.5l3 1.8"/>',
  pause: '<path d="M9 6.5v11M15 6.5v11"/>',
};

const svg = (a: Act) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[a]}</svg>`;

// Layout in units of 1vmin, capped so tablets get thumb-sized controls rather than huge ones
// (`--u` in style.css). Buttons are placed by their centre from the bottom-right corner of the
// safe area; `l` ones from the bottom-left, `t` ones from the top-left.
const UNIT_MAX = 4.6; // px
const BUTTONS: { act: Act; label: string; x: number; y: number; s: number; side?: 'l' | 't' }[] = [
  { act: 'fire', label: 'Fire', x: 17, y: 20, s: 21 },
  { act: 'aim', label: 'Aim', x: 35, y: 36, s: 15 },
  { act: 'reload', label: '', x: 38, y: 13, s: 13 },
  { act: 'jump', label: 'Jump', x: 12, y: 42, s: 12.5 },
  { act: 'crouch', label: 'Crouch', x: 55, y: 9.5, s: 12.5 },
  { act: 'light', label: 'Light', x: 12, y: 59.5, s: 11.5 },
  { act: 'fire', label: 'Fire', x: 12, y: 53, s: 13, side: 'l' },
  { act: 'pause', label: '', x: 8, y: 8, s: 11, side: 't' },
];
const STICK_R = 13; // ring radius, units

const DEAD = 0.12; // stick dead zone, fraction of the ring
const SPRINT_PAST = 1.25; // push this far past the centre (in ring radii), towards the top, to sprint
const FOLLOW = 1.7; // the stick base trails a thumb that wanders further than this

// A short buzz on each shot where the phone has it (Android). Chrome refuses it, with a console
// warning every time, inside a cross-origin frame such as the hosted build's viewer.
const HAPTICS = (() => {
  try {
    return typeof navigator.vibrate === 'function' && typeof window.top?.location.href === 'string';
  } catch {
    return false;
  }
})();

export class TouchControls {
  readonly el: HTMLDivElement;
  /** Look speed relative to the mouse: 1 px of drag = this many mouse px. */
  lookScale = 1.35;
  private pointers = new Map<number, Role>();
  private stick: HTMLDivElement;
  private knob: HTMLDivElement;
  private moveId: number | null = null;
  private base = { x: 0, y: 0 };
  private btn = new Map<Act, HTMLElement[]>();
  private ammoEl: HTMLElement;
  private aimFactor = 1;

  constructor(
    parent: HTMLElement,
    private input: Input,
    private opts: { onPause: () => void; allowMouse?: boolean },
  ) {
    const el = (this.el = document.createElement('div'));
    el.className = 'touch';
    el.hidden = true;
    this.stick = document.createElement('div');
    this.stick.className = 'stick';
    this.stick.innerHTML = '<div class="knob"></div><span class="sprint">Sprint</span>';
    this.knob = this.stick.firstElementChild as HTMLDivElement;
    el.append(this.stick);
    for (const b of BUTTONS) {
      const d = document.createElement('div');
      d.className = `tbtn ${b.act}${b.side ? ` ${b.side}` : ''}`;
      d.dataset.act = b.act;
      d.style.setProperty('--x', `${b.x}`);
      d.style.setProperty('--y', `${b.y}`);
      d.style.setProperty('--s', `${b.s}`);
      d.setAttribute('role', 'button');
      d.setAttribute('aria-label', b.act === 'reload' ? 'Reload' : b.act === 'pause' ? 'Pause' : b.label);
      d.innerHTML = svg(b.act) + (b.act === 'reload' ? '<span class="n"></span>' : b.label ? `<span>${b.label}</span>` : '');
      el.append(d);
      this.btn.set(b.act, [...(this.btn.get(b.act) ?? []), d]);
    }
    this.ammoEl = el.querySelector('.reload .n')!;
    parent.append(el);

    el.addEventListener('pointerdown', this.down);
    el.addEventListener('pointermove', this.move);
    el.addEventListener('pointerup', this.up);
    el.addEventListener('pointercancel', this.up);
    el.addEventListener('lostpointercapture', this.up);
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    window.addEventListener('resize', () => this.park());
    window.addEventListener('blur', () => this.release());
    document.addEventListener('visibilitychange', () => document.hidden && this.release());
    this.park();
  }

  show(on: boolean) {
    this.el.hidden = !on;
    if (!on) this.release();
  }

  /** Mirror the game state on the buttons. Call once a frame. */
  update(h: TouchHud) {
    this.aimFactor = 1 - 0.45 * h.aim;
    this.ammoEl.textContent = h.reloading ? '···' : `${h.ammo}/${h.reserve}`;
    this.set('reload', 'on', h.reloading || (h.ammo === 0 && h.reserve > 0));
    this.set('aim', 'on', this.input.aim);
    this.set('crouch', 'on', this.input.crouch);
    this.set('light', 'on', h.light);
    this.stick.classList.toggle('sprinting', this.input.sprint);
  }

  /** Let go of everything (pause, lost focus). */
  release() {
    for (const id of [...this.pointers.keys()]) this.end(id);
    this.input.fire = false;
  }

  private set(act: Act, cls: string, on: boolean) {
    for (const b of this.btn.get(act) ?? []) b.classList.toggle(cls, on);
  }

  private get radius() {
    return STICK_R * Math.min(Math.min(innerWidth, innerHeight) / 100, UNIT_MAX);
  }

  /** The resting stick, bottom left where the thumb is expected (placed by style.css). */
  private park() {
    if (this.moveId !== null) return;
    this.stick.style.left = this.stick.style.top = '';
    this.stick.classList.add('parked');
    this.stick.classList.remove('active', 'sprinting');
    this.knob.style.transform = '';
  }

  private place(x: number, y: number) {
    this.base = { x, y };
    this.stick.classList.remove('parked');
    this.stick.style.left = `${x}px`;
    this.stick.style.top = `${y}px`;
  }

  private down = (e: PointerEvent) => {
    if (e.pointerType === 'mouse' && (!this.opts.allowMouse || e.button !== 0)) return;
    // No emulated mouse events, text selection, magnifier or scrolling from this touch.
    e.preventDefault();
    this.el.setPointerCapture(e.pointerId);
    const x = e.clientX;
    const y = e.clientY;
    const b = (e.target as HTMLElement).closest<HTMLElement>('[data-act]');
    if (b) {
      const act = b.dataset.act as Act;
      b.classList.add('down');
      if (act === 'fire') {
        this.pointers.set(e.pointerId, { kind: 'fire', x, y, btn: b });
        this.input.fire = true;
        this.input.firePressed = true;
        if (HAPTICS) navigator.vibrate(8);
        return;
      }
      this.pointers.set(e.pointerId, { kind: 'button', btn: b, act });
      if (act === 'aim') this.input.aim = !this.input.aim;
      else if (act === 'reload') this.input.press('KeyR');
      else if (act === 'jump') this.input.press('Space');
      else if (act === 'light') this.input.press('KeyF');
      else if (act === 'crouch') this.input.crouch = !this.input.crouch;
      return;
    }
    if (x < innerWidth * 0.45 && this.moveId === null) {
      // The stick starts under the thumb (kept clear of the screen edge).
      const r = this.radius;
      this.moveId = e.pointerId;
      this.pointers.set(e.pointerId, { kind: 'move' });
      this.place(Math.max(x, r + 4), Math.min(y, innerHeight - r - 4));
      this.stick.classList.add('active');
      this.steer(x, y);
      return;
    }
    this.pointers.set(e.pointerId, { kind: 'look', x, y });
  };

  private move = (e: PointerEvent) => {
    const r = this.pointers.get(e.pointerId);
    if (!r) return;
    if (r.kind === 'move') this.steer(e.clientX, e.clientY);
    else if (r.kind === 'look' || r.kind === 'fire') {
      const k = this.lookScale * this.aimFactor;
      this.input.look((e.clientX - r.x) * k, (e.clientY - r.y) * k);
      r.x = e.clientX;
      r.y = e.clientY;
    }
  };

  private up = (e: PointerEvent) => this.end(e.pointerId);

  private end(id: number) {
    const r = this.pointers.get(id);
    if (!r) return;
    this.pointers.delete(id);
    if (this.el.hasPointerCapture(id)) this.el.releasePointerCapture(id);
    if (r.kind === 'move') {
      this.moveId = null;
      this.input.moveX = this.input.moveY = 0;
      this.input.sprint = false;
      this.park();
    } else if (r.kind === 'fire') {
      r.btn.classList.remove('down');
      if (![...this.pointers.values()].some((p) => p.kind === 'fire')) this.input.fire = false;
    } else if (r.kind === 'button') {
      r.btn.classList.remove('down');
      // On release, so the same tap can't land on the pause menu and resume straight away.
      if (r.act === 'pause') this.opts.onPause();
    }
  }

  private steer(x: number, y: number) {
    const R = this.radius;
    let dx = x - this.base.x;
    let dy = y - this.base.y;
    let d = Math.hypot(dx, dy);
    if (d > FOLLOW * R) {
      // Drag the base along behind a wandering thumb.
      const k = (d - FOLLOW * R) / d;
      this.place(this.base.x + dx * k, this.base.y + dy * k);
      dx = x - this.base.x;
      dy = y - this.base.y;
      d = Math.hypot(dx, dy);
    }
    const c = Math.min(1, R / Math.max(d, 1e-6));
    this.knob.style.transform = `translate(${dx * c}px, ${dy * c}px)`;
    // Analog value with a dead zone, within the unit circle.
    const m = Math.min(1, d / R);
    const g = m < DEAD || d < 1e-6 ? 0 : (m - DEAD) / (1 - DEAD) / d;
    this.input.moveX = dx * g;
    this.input.moveY = -dy * g;
    const sprint = d > SPRINT_PAST * R && -dy > Math.abs(dx) * 1.2;
    if (sprint && !this.input.sprint) {
      // Sprinting stands you up and lowers the gun, as on the keyboard.
      this.input.crouch = false;
      this.input.aim = false;
    }
    this.input.sprint = sprint;
  }
}
