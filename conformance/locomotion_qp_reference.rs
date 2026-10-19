//! One-step convex-quadratic translational MPC baseline (not full 22-agent
//! nonlinear physics). Units: m, m/s, m/s², seconds. Candidate before solver
//! benchmarking and physically grounded yaw/contact constraints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion { pub x:f64, pub y:f64, pub vx:f64, pub vy:f64 }
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Control { pub ax:f64,pub ay:f64 }
#[derive(Clone, Copy, Debug)]
pub struct Weights { pub position:f64,pub velocity:f64,pub effort:f64 }
#[derive(Clone, Copy, Debug, PartialEq,Eq)]
pub enum PlanError { NonFinite,InvalidConstraints }
fn axis(position:f64, velocity:f64, target:f64,target_v:f64,
    dt:f64,w:Weights,a_max:f64)->f64{
    let pa=0.5*dt*dt;
    let numerator=w.position*(position+dt*velocity-target)*pa
        +w.velocity*(velocity-target_v)*dt;
    let denom=w.position*pa*pa+w.velocity*dt*dt+w.effort;
    (-numerator/denom).clamp(-a_max,a_max)
}
pub fn solve_one_step(state:Motion,target:[f64;2],target_v:[f64;2],
 dt:f64,weights:Weights,max_acceleration:f64)->Result<Control,PlanError>{
 let all=[state.x,state.y,state.vx,state.vy,target[0],target[1],target_v[0],target_v[1],
  dt,weights.position,weights.velocity,weights.effort,max_acceleration];
 if all.iter().any(|v|!v.is_finite()){return Err(PlanError::NonFinite);}
 if dt<=0.0||dt>1.0||max_acceleration<=0.0||
   weights.position<0.0||weights.velocity<0.0||weights.effort<=0.0 {
    return Err(PlanError::InvalidConstraints);
 }
 Ok(Control{
  ax:axis(state.x,state.vx,target[0],target_v[0],dt,weights,max_acceleration),
  ay:axis(state.y,state.vy,target[1],target_v[1],dt,weights,max_acceleration),
 })
}
pub fn advance(state:Motion,control:Control,dt:f64)->Motion{
 Motion{x:state.x+state.vx*dt+0.5*control.ax*dt*dt,
 y:state.y+state.vy*dt+0.5*control.ay*dt*dt,
 vx:state.vx+control.ax*dt,vy:state.vy+control.ay*dt}
}
#[cfg(test)]mod tests{
 use super::*;
 fn weights()->Weights{Weights{position:10.0,velocity:1.0,effort:0.01}}
 #[test]fn deterministic_solution_respects_bounded_control(){
  let state=Motion{x:0.,y:0.,vx:0.,vy:0.};
  let a=solve_one_step(state,[100.,0.],[0.,0.],0.1,weights(),2.5).unwrap();
  let b=solve_one_step(state,[100.,0.],[0.,0.],0.1,weights(),2.5).unwrap();
  assert_eq!(a,b);assert!(a.ax<=2.5&&a.ax>0.0);assert_eq!(a.ay,0.0);
  let next=advance(state,a,0.1);assert!(next.x>0.0);
 }
 #[test]fn optimal_constrained_solution_beats_zero_acceleration_in_one_step(){
  let s=Motion{x:0.,y:0.,vx:0.,vy:0.};
  let c=solve_one_step(s,[10.,0.],[1.,0.],0.25,weights(),4.).unwrap();
  let cost=|c:Control|{
   let n=advance(s,c,0.25);
   weights().position*(n.x-10.).powi(2)+weights().velocity*(n.vx-1.).powi(2)
    +weights().effort*c.ax.powi(2)
  };
  assert!(cost(c)<cost(Control{ax:0.,ay:0.}));
 }
 #[test]fn invalid_nonconvex_or_nonsensical_inputs_fail_closed(){
  let s=Motion{x:0.,y:0.,vx:0.,vy:0.};
  assert_eq!(solve_one_step(s,[1.,0.],[0.,0.],0.,weights(),2.),Err(PlanError::InvalidConstraints));
  assert_eq!(solve_one_step(s,[f64::NAN,0.],[0.,0.],0.1,weights(),2.),Err(PlanError::NonFinite));
  assert_eq!(solve_one_step(s,[1.,0.],[0.,0.],0.1,weights(),-1.),Err(PlanError::InvalidConstraints));
 }
}
