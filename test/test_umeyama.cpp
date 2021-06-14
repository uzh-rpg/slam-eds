/*
* This file is part of the EDS: Event-aided Direct Sparse Odometry
* (https://rpg.ifi.uzh.ch/eds.html)
*
* Copyright (c) 2022 Javier Hidalgo-Carrió, Robotics and Perception
* Group (RPG) University of Zurich.
*
* EDS is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* EDS is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/test/unit_test.hpp>
#include <eds/EDS.h>
#include <Eigen/Geometry>
#include <Eigen/LU> // required for MatrixBase::determinant
#include <Eigen/SVD> // required for SVD

#include <yaml-cpp/yaml.h>
#include <iostream>
using namespace eds;


//  Constructs a random matrix from the unitary group U(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixUnitary(int size)
{
  typedef T Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  MatrixType Q;

  int max_tries = 40;
  double is_unitary = false;

  while (!is_unitary && max_tries > 0)
  {
    // initialize random matrix
    Q = MatrixType::Random(size, size);

    // orthogonalize columns using the Gram-Schmidt algorithm
    for (int col = 0; col < size; ++col)
    {
      typename MatrixType::ColXpr colVec = Q.col(col);
      for (int prevCol = 0; prevCol < col; ++prevCol)
      {
        typename MatrixType::ColXpr prevColVec = Q.col(prevCol);
        colVec -= colVec.dot(prevColVec)*prevColVec;
      }
      Q.col(col) = colVec.normalized();
    }

    // this additional orthogonalization is not necessary in theory but should enhance
    // the numerical orthogonality of the matrix
    for (int row = 0; row < size; ++row)
    {
      typename MatrixType::RowXpr rowVec = Q.row(row);
      for (int prevRow = 0; prevRow < row; ++prevRow)
      {
        typename MatrixType::RowXpr prevRowVec = Q.row(prevRow);
        rowVec -= rowVec.dot(prevRowVec)*prevRowVec;
      }
      Q.row(row) = rowVec.normalized();
    }

    // final check
    is_unitary = Q.isUnitary();
    --max_tries;
  }

  if (max_tries == 0)
    eigen_assert(false && "randMatrixUnitary: Could not construct unitary matrix!");

  return Q;
}

//  Constructs a random matrix from the special unitary group SU(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixSpecialUnitary(int size)
{
  typedef T Scalar;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixType;

  // initialize unitary matrix
  MatrixType Q = randMatrixUnitary<Scalar>(size);

  // tweak the first column to make the determinant be 1
  Q.col(0) *= Eigen::numext::conj(Q.determinant());

  return Q;
}

template<typename Scalar, int Dimension>
void run_fixed_size_test(int num_elements)
{
  BOOST_TEST_MESSAGE("###### TEST_UMEYAMA WITH "<< num_elements <<" #ELEMENTS ######");
  using std::abs;
  typedef Eigen::Matrix<Scalar, Dimension+1, Eigen::Dynamic> MatrixX;
  typedef Eigen::Matrix<Scalar, Dimension+1, Dimension+1> HomMatrix;
  typedef Eigen::Matrix<Scalar, Dimension, Dimension> FixedMatrix;
  typedef Eigen::Matrix<Scalar, Dimension, 1> FixedVector;

  const int dim = Dimension;

  // MUST be positive because in any other case det(cR_t) may become negative for
  // odd dimensions!
  // Also if c is to small compared to t.norm(), problem is ill-posed (cf. Bug 744)
  const Scalar c = Eigen::internal::random<Scalar>(0.5, 2.0);

  FixedMatrix R = randMatrixSpecialUnitary<Scalar>(dim);
  FixedVector t = Scalar(32)*FixedVector::Random(dim,1);

  HomMatrix cR_t = HomMatrix::Identity(dim+1,dim+1);
  cR_t.block(0,0,dim,dim) = c*R;
  cR_t.block(0,dim,dim,1) = t;

  MatrixX src = MatrixX::Random(dim+1, num_elements);
  src.row(dim) = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>::Constant(num_elements, Scalar(1));

  MatrixX dst = cR_t*src;
  BOOST_TEST_MESSAGE("src size: "<<src.rows()<<" x "<<src.cols());
  BOOST_TEST_MESSAGE("dst size: "<<dst.rows()<<" x "<<dst.cols());
  BOOST_TEST_MESSAGE("cR_t: "<<cR_t.rows()<<" x "<<cR_t.cols());

  Eigen::Block<MatrixX, Dimension, Eigen::Dynamic> src_block(src,0,0,dim,num_elements);
  Eigen::Block<MatrixX, Dimension, Eigen::Dynamic> dst_block(dst,0,0,dim,num_elements);

  HomMatrix cR_t_umeyama = umeyama(src_block, dst_block);

  const Scalar error = ( cR_t_umeyama*src - dst ).squaredNorm();
  BOOST_TEST_MESSAGE("Error: "<<error);

  FixedMatrix C = cR_t_umeyama.block(0,0, dim,dim) * R.inverse();
  BOOST_TEST_MESSAGE("c: "<<c<<" c_estimate: "<<C.trace()/dim);
  BOOST_TEST_MESSAGE("R\n"<<R);
  BOOST_TEST_MESSAGE("cR\n"<<cR_t.block(0,0, dim, dim));
  BOOST_TEST_MESSAGE("cR_umeyama\n"<<cR_t_umeyama.block(0,0, dim, dim));

  BOOST_CHECK(error < Scalar(16)*std::numeric_limits<Scalar>::epsilon());
}


BOOST_AUTO_TEST_CASE(test_umeyama)
{
    const int num_elements = Eigen::internal::random<int>(40,500);
    run_fixed_size_test<float, 2>(num_elements);
    run_fixed_size_test<float, 3>(num_elements);
    run_fixed_size_test<double, 2>(num_elements);
    run_fixed_size_test<double, 3>(num_elements);
}

BOOST_AUTO_TEST_CASE(adhoc_test)
{
    BOOST_TEST_MESSAGE("###### AD-HOC TEST_UMEYAMA ######");
    const int num_elements = 100;

    const double c = Eigen::internal::random<double>(0.5, 2.0);
    ::base::Matrix3d R = randMatrixSpecialUnitary<double>(3);
    ::base::Vector3d t = double(32)*::base::Vector3d::Random(3,1);

    ::base::Matrix4d cR_t = ::base::Matrix4d::Identity();
    cR_t.block(0,0,3,3) = c*R;
    cR_t.block(0,3,3,1) = t;

    typedef Eigen::Matrix<double, 4, num_elements> MatrixX;
    MatrixX src = MatrixX::Random(4, num_elements);
    src.row(3) = Eigen::Matrix<double, 1, Eigen::Dynamic>::Constant(num_elements, double(1));

    MatrixX dst = cR_t*src;

    /** Compute Umeyama **/
    Eigen::Block<MatrixX, 3, num_elements> src_block(src,0,0,3,num_elements);
    Eigen::Block<MatrixX, 3, num_elements> dst_block(dst,0,0,3,num_elements);

    ::base::Matrix4d cR_t_umeyama = umeyama(src_block, dst_block);

    const double error = ( cR_t_umeyama*src - dst ).squaredNorm();
    BOOST_TEST_MESSAGE("Error: "<<error);

    ::base::Matrix3d C = cR_t_umeyama.block(0,0, 3, 3) * R.inverse();
    BOOST_TEST_MESSAGE("c: "<<c<<" c_estimate: "<<C.trace()/3);
    BOOST_TEST_MESSAGE("R\n"<<R);
    BOOST_TEST_MESSAGE("cR\n"<<cR_t.block(0, 0, 3, 3));
    BOOST_TEST_MESSAGE("cR_umeyama\n"<<cR_t_umeyama.block(0,0, 3, 3));

    BOOST_CHECK(error < double(16)*std::numeric_limits<double>::epsilon());

    BOOST_TEST_MESSAGE("cR_t\n"<<cR_t);
    eds::utils::Alignment<double, num_elements> my_align;
    ::base::Matrix4d cR_t_bis = ::base::Matrix4d::Identity();

    for (int i=0; i<num_elements; ++i)
    {
        Eigen::Vector3d model_vector; model_vector << src.col(i)[0], src.col(i)[1], src.col(i)[2];
        Eigen::Vector3d data_vector; data_vector << dst.col(i)[0], dst.col(i)[1], dst.col(i)[2];
        cR_t_bis = my_align.realign(model_vector, data_vector);
    }
    BOOST_TEST_MESSAGE("cR_t_bis\n"<<cR_t_bis);
    const double error_bis = ( cR_t_bis*src - dst ).squaredNorm();
    BOOST_CHECK(error_bis < double(16)*std::numeric_limits<double>::epsilon());
}