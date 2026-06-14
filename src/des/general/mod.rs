//! `des::general` mirror: re-export the generic des_engine modules the soccer
//! code imports, plus the local soccer modules.

pub use des_engine::des::general::{
    des_base, general, hungarian, ip_mip_des, lp, lp_des, mpc_point_mass, neural_network, prng, qp,
};

pub mod soccer;
pub mod soccer_rotation;
pub mod tournament;
