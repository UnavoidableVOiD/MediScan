import { Component, Suspense, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Environment, Float, Sparkles } from "@react-three/drei";
import * as THREE from "three";

/* ------------------------------------------------------------------
   A slowly turning double helix of frosted glass beads with a soft
   emerald light rig — reads as "DNA / lab", stays inside green + white.
   ------------------------------------------------------------------ */

/* ------------------------------------------------------------------
   Graceful degradation: the 3D layer is decorative. If WebGL is missing
   (old devices, disabled GPU, headless) we render nothing rather than
   let a renderer error take the page down.
   ------------------------------------------------------------------ */
function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return Boolean(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

function useWebGL() {
  // lazy initializer: evaluated once on the client, never during SSR
  const [ok] = useState<boolean>(() => typeof document !== "undefined" && hasWebGL());
  return ok;
}

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

const BEADS = 34;
const RADIUS = 1.35;
const HEIGHT = 7.5;

function Helix({ mouse }: { mouse: React.MutableRefObject<{ x: number; y: number }> }) {
  const group = useRef<THREE.Group>(null);
  const beads = useMemo(() => {
    const out: { pos: THREE.Vector3; pos2: THREE.Vector3; s: number }[] = [];
    for (let i = 0; i < BEADS; i++) {
      const t = i / (BEADS - 1);
      const a = t * Math.PI * 4.2;
      const y = (t - 0.5) * HEIGHT;
      out.push({
        pos: new THREE.Vector3(Math.cos(a) * RADIUS, y, Math.sin(a) * RADIUS),
        pos2: new THREE.Vector3(Math.cos(a + Math.PI) * RADIUS, y, Math.sin(a + Math.PI) * RADIUS),
        s: 0.16 + 0.06 * Math.sin(t * Math.PI),
      });
    }
    return out;
  }, []);

  useFrame((state, dt) => {
    if (!group.current) return;
    group.current.rotation.y += dt * 0.22;
    const targetX = mouse.current.y * 0.25;
    const targetZ = mouse.current.x * 0.18;
    group.current.rotation.x = THREE.MathUtils.lerp(group.current.rotation.x, targetX, 0.04);
    group.current.rotation.z = THREE.MathUtils.lerp(group.current.rotation.z, targetZ, 0.04);
    group.current.position.y = Math.sin(state.clock.elapsedTime * 0.6) * 0.15;
  });

  const glass = (
    <meshPhysicalMaterial
      color="#d1fae5"
      roughness={0.12}
      metalness={0}
      transmission={0.92}
      thickness={0.8}
      ior={1.4}
      clearcoat={1}
      clearcoatRoughness={0.1}
      attenuationColor="#34d399"
      attenuationDistance={1.6}
    />
  );
  const solid = (
    <meshStandardMaterial
      color="#059669"
      roughness={0.3}
      metalness={0.1}
      emissive="#065f46"
      emissiveIntensity={0.25}
    />
  );

  return (
    <group ref={group} rotation={[0.35, 0, -0.35]}>
      {beads.map((b, i) => (
        <group key={i}>
          <mesh position={b.pos} castShadow>
            <sphereGeometry args={[b.s, 32, 32]} />
            {i % 3 === 0 ? solid : glass}
          </mesh>
          <mesh position={b.pos2}>
            <sphereGeometry args={[b.s * 0.9, 32, 32]} />
            {i % 3 === 1 ? solid : glass}
          </mesh>
          {/* rung */}
          <Rung a={b.pos} b={b.pos2} />
        </group>
      ))}
    </group>
  );
}

function Rung({ a, b }: { a: THREE.Vector3; b: THREE.Vector3 }) {
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const dir = b.clone().sub(a);
  const len = dir.length();
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.clone().normalize(),
  );
  return (
    <mesh position={mid} quaternion={quat}>
      <cylinderGeometry args={[0.018, 0.018, len, 8]} />
      <meshStandardMaterial color="#a7f3d0" transparent opacity={0.55} roughness={0.5} />
    </mesh>
  );
}

function Orbs() {
  const items = useMemo(
    () =>
      Array.from({ length: 7 }, (_, i) => ({
        p: [Math.sin(i * 1.7) * 3.2, Math.cos(i * 2.3) * 2.4, -1.5 - (i % 3)] as [
          number,
          number,
          number,
        ],
        r: 0.25 + (i % 3) * 0.12,
        speed: 0.6 + (i % 4) * 0.2,
      })),
    [],
  );
  return (
    <>
      {items.map((o, i) => (
        <Float key={i} speed={o.speed} rotationIntensity={0.2} floatIntensity={1.4}>
          <mesh position={o.p}>
            <sphereGeometry args={[o.r, 32, 32]} />
            <meshPhysicalMaterial
              color="#ecfdf5"
              transmission={0.95}
              roughness={0.05}
              thickness={1.2}
              ior={1.3}
              attenuationColor="#6ee7b7"
              attenuationDistance={2}
            />
          </mesh>
        </Float>
      ))}
    </>
  );
}

function Scene() {
  const mouse = useRef({ x: 0, y: 0 });
  useFrame(({ pointer }) => {
    mouse.current.x = pointer.x;
    mouse.current.y = pointer.y;
  });
  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[4, 6, 5]} intensity={1.6} color="#ffffff" />
      <pointLight position={[-5, -2, 3]} intensity={12} color="#34d399" />
      <pointLight position={[3, -4, -2]} intensity={8} color="#14b8a6" />
      <Float speed={1} rotationIntensity={0.15} floatIntensity={0.6}>
        <Helix mouse={mouse} />
      </Float>
      <Orbs />
      <Sparkles
        count={90}
        scale={[10, 8, 6]}
        size={2.2}
        speed={0.35}
        color="#10b981"
        opacity={0.55}
      />
      <Environment preset="city" environmentIntensity={0.5} />
    </>
  );
}

export function HeroScene({ className }: { className?: string }) {
  const gl = useWebGL();
  if (!gl) return null;
  return (
    <div className={className} aria-hidden>
      <SceneBoundary>
        <Canvas
          dpr={[1, 1.6]}
          camera={{ position: [0, -0.3, 10.5], fov: 38 }}
          gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
          style={{ background: "transparent" }}
        >
          <Suspense fallback={null}>
            <Scene />
          </Suspense>
        </Canvas>
      </SceneBoundary>
    </div>
  );
}

/* A lighter single-orb scene for auth and empty states. */
export function OrbScene({ className }: { className?: string }) {
  const gl = useWebGL();
  if (!gl) return null;
  return (
    <div className={className} aria-hidden>
      <SceneBoundary>
        <Canvas
          dpr={[1, 1.5]}
          camera={{ position: [0, 0, 5], fov: 40 }}
          gl={{ alpha: true }}
          style={{ background: "transparent" }}
        >
          <Suspense fallback={null}>
            <ambientLight intensity={0.9} />
            <directionalLight position={[3, 4, 5]} intensity={1.5} />
            <pointLight position={[-4, -2, 2]} intensity={10} color="#34d399" />
            <Float speed={1.4} rotationIntensity={0.4} floatIntensity={1}>
              <mesh>
                <icosahedronGeometry args={[1.35, 4]} />
                <meshPhysicalMaterial
                  color="#d1fae5"
                  transmission={0.9}
                  roughness={0.08}
                  thickness={1.4}
                  ior={1.35}
                  clearcoat={1}
                  attenuationColor="#10b981"
                  attenuationDistance={1.8}
                />
              </mesh>
            </Float>
            <Sparkles
              count={40}
              scale={[6, 6, 4]}
              size={2}
              speed={0.3}
              color="#10b981"
              opacity={0.5}
            />
            <Environment preset="city" environmentIntensity={0.5} />
          </Suspense>
        </Canvas>
      </SceneBoundary>
    </div>
  );
}
